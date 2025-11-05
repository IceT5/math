## Kernel实现

### Kernel简介
Kernel是算子在NPU执行的核心部分，负责张量数据的加载、计算和存储，是算子功能实现的最终载体。Kernel的实现需要与Tiling策略紧密配合，根据Tiling提供的`TilingData`、`TilingKey`信息进行内存分配和计算调度。

Kernel实现包括如下步骤，整个流程通过`Process`函数串联，实现完整的算子流程。

```mermaid
graph LR
	H([核函数定义]) -->A([定义Kernel类])
	A -->B([初始化函数<br>Init])
    B --> C([主处理函数<br>Process])
    C -->D([数据搬入<br>CopyIn])
    D -->E([计算<br>Compute])
    E -->F([数据搬出<br>CopyOut])
    F -->G([Kernel执行完成])
```

### 代码实现

Kernel一个需要两个交付件：`${op_name}.cpp` `${op_name}.h`

Kernel入口文件\$\{op\_name\}.cpp ，包含主函数和调度逻辑，`AddExample`算子完整代码请参考`examples/add_example/op_kernel`下[add_example.cpp](../../examples/add_example/op_kernel/add_example.cpp)。

\$\{op\_name\}.h中定义Kernel头文件，包含函数声明、结构定义、逻辑实现等，`AddExample`算子完整代码请参考`examples/add_example/op_kernel`下[add_example.h](../../examples/add_example/op_kernel/add_example.h)。

## aclnn适配

通常算子开发和编译完成后，会自动生成aclnn接口（一套基于C 的API），可直接在应用程序中调用aclnn接口实现调用算子。

为实现该调用方式，需提前生成算子对应的二进制包，增加二进制编译json文件，以`AddExample`算子为例：

1. 在`examples/add_example/op_host`目录新建`config/${soc_version}`文件夹，用于存放配置文件。

2. 在`${soc_version}`目录新建json文件，命名为`${op_name}_binary.json`，用于描述算子相关信息，包括二进制文件名称(命名无要求，当前是以`${op_type}`_哈希码命名)及算子输入、输出、shape、data type、format等信息，完整定义请参考[add_example_binary.json](../../examples/add_example/op_host/config/ascend910b/add_example_binary.json)。

3. 在`${soc_version}`目录新建ini文件，命名为`${op_name}_simplified_key.ini`，与二进制匹配逻辑相关，默认是0，示例参考[add_example_simplified_key.ini](../../examples/add_example/op_host/config/ascend910b/add_example_simplified_key.ini)。

4. 在`scripts/kernel/binary_config`目录[ascendc_config.json](../../scripts/kernel/binary_config/ascendc_config.json)中，注册算子的NPU型号和实现模式，示例如下，输入实际name和compute_units即可。

    ```json
    {"name":"AddExample", "compute_units": ["${soc_version}"], "auto_sync":true, "impl_mode" : "high_performance"},
    ```

## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包\*\.run，详细的编译操作如下：

1. **准备工作。**

    参考[前提条件](#前提条件)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。

2. **编译自定义算子包。**

    以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example](../../examples/add_example)目录。

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译指定算子，如--ops=add_example
    bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list}
    ```

    若提示如下信息，说明编译成功：

    ```bash
    Self-extractable archive "cann-ops-math-${vendor_name}_linux-${arch}.run" successfully created.
    ```

3. **安装自定义算子包。**

    执行以下命令进行安装：
    
    ```bash
    ./cann-ops-math-${vendor_name}_linux-${arch}.run
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/latest/opp/vendors/${vendor_name}/op_api/lib:${LD_LIBRARY_PATH}
    ```
    自定义算子包安装在`${ASCEND_HOME_PATH}/latest/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录，可提前在环境变量中配置。自定义算子包不支持卸载。
    
    自定义算子包的目录结构示例如下：
    ```
    ├── cann-ops-math-${vendor_name}_linux-${arch}.run           # 包名
    ├── bin
    │   └── set_env.bash                                         # 环境变量source脚本
    ├── op_api
    │   ├── include
    │   │   ├── aclnn_add_example.h                              # aclnn头文件
    │   │   └── aclnn_ops_math_${vendor_name}.h                  # aclnn汇总头文件
    │   └── lib
    │       └── libcust_opapi.so                                 # 算子aclnn接口动态库
    ├── op_impl
    │   └── ai_core
    │       └── tbe
    │           ├── config
    │           │   └── ${soc_version}
    │           │       └── aic-${soc_version}-ops-info.json     # 算子信息库
    │           ├── custom_impl
    │           │   ├── ascendc
    │           │   │   ├── add_example
    │           │   ├── add_example.cpp                          # Kernel实现
    │           │   │   ├── add_example.h
    │           │   │   ├── add_example_tiling_data.h
    │           │   │   └── add_example_tiling_key.h
    │           │   └── dynamic
    │           │       └── add_example.py
    │           ├── kernel
    │           │   ├── ${soc_version}                     
    │           │   │   └── add_example                         # 算子二进制文件
    │           │   │       ├── AddExample_11132827238e1555db7b997c7bce2928_high_performance.json
    │           │   │       ├── AddExample_11132827238e1555db7b997c7bce2928_high_performance.o
    │           │   │       ├── AddExample_a1532827238e1555db7b997c7bce2928_high_performance.json
    │           │   │       └── AddExample_a1532827238e1555db7b997c7bce2928_high_performance.o
    │           │   └── config
    │           │       └── ${soc_version}                     # 算子二进制配置
    │           │           ├── add_example.json
    │           │           └── binary_info_config.json
    │           └── op_tiling                                  # Tiling实现
    │               ├── lib
    │               │   └── linux
    │               │           └── ${arch}
    │               │               └── libcust_opmaster_rt2.0.so
    │               └── liboptiling.so -> lib/linux/${arch}/libcust_opmaster_rt2.0.so
    ├── op_proto
    │   ├── inc
    │   │   └── add_example_proto.h
    │   └── lib
    │       └── linux
    │           └── ${arch}
    │               └── libcust_opsproto_rt2.0.so
    └── version.info                                           # 包信息
    ```

## 算子验证

1. **UT验证。**

    算子开发过程中，可通过UT验证（如kernel UT，aclnn UT）方式进行快速验证，方法请参考[算子调用方式](../invocation/quick_op_invocation.md)。

2. **aclnn调用验证。**

    开发好的算子完成编译部署后，可通过aclnn方式验证功能，方法请参考[算子调用方式](../invocation/op_invocation.md)。

## 附录

自定义算子如需运行图模式，不需要aclnn适配，做如下交付件适配，详细内容请参考[图模式开发指南](./graph_develop_guide.md)。
