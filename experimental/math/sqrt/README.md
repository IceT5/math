# Sqrt

## 目录结构介绍

```
sqrt                                    # Sqrt算子目录 
├── docs                                # Sqrt算子文档目录 
├── op_host                             # Host侧实现 
│   ├── sqrt_def.cpp                    # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等 
│   ├── sqrt_infershape.cpp             # InferShape实现，实现算子形状推导，在运行时推导输出shape 
│   ├── sqrt_tiling.cpp                 # Tiling实现，将张量划分为多个小块，区分数据类型进行并行计算 
│   └── CMakeLists.txt                  # Host侧cmakelist文件 
└── op_kernel                           # Device侧Kernel实现 
│   ├── sqrt_tiling_key.h               # Tilingkey文件，定义Tiling策略的Key，标识不同的划分方式 
│   ├── sqrt_tiling_data.h              # Tilingdata文件，存储Tiling策略相关的配置数据，如块大小、并行度 
│   ├── sqrt.cpp                        # Kernel入口文件，包含主函数和调度逻辑 
│   └── sqrt.h                          # Kernel实现文件，定义Kernel头文件，包含函数声明、结构定义、逻辑实现 
├── op_graph                            # 图融合相关实现 
│   ├── CMakeLists.txt                  # op_graph侧cmakelist文件 
│   ├── sqrt_graph_infer.cpp            # InferDataType文件，实现算子类型推导，在运行时推导输出dataType 
│   └── sqrt_proto.h                    # 算子原型定义，用于图优化和融合阶段识别算子 
├── test                                # 测试目录 
│   ├── CMakeLists.txt                  # 算子测试cmakelist入口 
│   └── ut                              # ut测试目录
│       ├── op_host                     # Host侧测试文件目录 
│       ├── op_kernel                   # Device侧测试文件目录 
│       └── CMakeLists.txt              # 算子ut测试cmakelist入口 
└── CMakeLists.txt                      # 算子cmakelist入口 
```

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Sqrt`算子返回输入数据经过开方运算的结果。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Sqrt</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
     
    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sqrt</td></tr>  
  </table>

## 约束与限制

- x，y，out的数据类型仅支持float32,float16,bfloat16，数据格式仅支持ND

## 算子使用
使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署
  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --experimental --soc=ascend910b --ops=sqrt
    ```

  - 部署算子包

    ```bash
    bash build_out/cann-ops-<vendor_name>-linux.<arch>.run
    ```
### 算子调用
    
    ```bash
    bash build.sh --run_example sqrt eager cust --vendor_name=custom
    ```