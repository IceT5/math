### Backgroud（背景信息）
            
#### Atan算子实现优化
基于Atan算子历史TBE版本使用Ascend C编程语言进行优化。

#### Atan算子现状分析
通过对Atan算子TBE版本的功能分析，当前支持的能力如下：
- 支持float16，float32两种格式的输入。
- 注册和参数检查阶段声称支持bfloat16，但在实际计算逻辑中缺少相应的类型转换处理。
- Atan算子不涉及到对输入数据进行广播（输入数据的shape调整到相同大小）

Atan算子TBE版本的整体流程图如下图所示：
![alt text](atan-tbe.png)
atan_compute接口具体实现如下图所示：
![alt text](atan_compute.png)
_do_taylor接口具体实现如下图所示：
![alt text](_do_taylor.png)

### Benefit / Necessity （价值/作用）
            
实现了Atan算子的AscendC实现，替代原有TBE算子在昇腾硬件上的适配,增加了bfloat16类型适配

算子原型
| 名称 | 类别 | dtype | format |shape |
|------|------|:------:|------|:----:|
| x | 输入 | fp16/fp32/bfp16 | ND |all |
| y | 输出 | fp16/fp32/bfp16 | ND | 同输入 |

算子支持型号
Atlas A2 训练系列产品

### Design（设计方案）
            
#  一、host侧设计：
## tiling策略：
Atan算子输入不需要广播，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。
任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。
批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

## 分核策略
优先使用满核的原则。
如果核间能均分，可视作无大小核区分，大核小核数据块一致；
如果核间不能均分，需要将余出的数据块分配到前几个核上。
输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。
UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

## 单core内切分策略
充分使用UB空间的原则。
需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。
UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。
Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM和不同类型下的ubDataNum，计算出每个Tile块的数据数量。
数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。
设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到ErfTilingData对象中。
这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

## tilingkey规划策略
不进行tilingkey划分，在kernel侧利用输入数据的类型来走不同的分支。

## kernel侧设计：
进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。将float16、bfloat16的数据都转成float32进行计算，其余数据类型保持原类型计算。
![alt text](last.png)

