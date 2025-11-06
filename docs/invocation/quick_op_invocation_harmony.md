# 算子调用
> **说明1**：本项目阐述如何使用鸿蒙社区版CANN开发套件包完成面向鸿蒙设备的算子调用。
>
> **说明2**：本项目可调用的算子参见[鸿蒙算子支持列表](../op_list_harmony.md)。


## 前提条件

使用本项目前，请确保如下基础依赖已安装。

**安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - Gawk
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
   ```bash
   bash install_deps.sh
   ```

## 环境准备

1. **安装鸿蒙社区版CANN开发套件包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **配置环境变量**
	
	根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/set_env.sh
   # 指定路径安装
   # source ${install_path}/set_env.sh
    ```

3. **下载源码**

    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/ops-math.git
    # 安装根目录requirements.txt依赖
    pip3 install -r requirements.txt
    ```

## 编译执行

若基于社区版CANN包对算子源码修改，可使用[自定义算子包](#自定义算子包)和[ops-math包](#ops-math包)方式编译执行。

- 自定义算子包：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。注意自定义算子包优先级高于原始CANN包。
- ops-math包：选择整个项目编译生成的包称为ops-math包，可**完整替换**CANN包对应部分。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以Abs算子编译为例
    # bash build.sh --pkg --soc=kirinx90 --ops=abs
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。鸿蒙Hardon PC / Hooper PC 使用"kirinx90"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"abs,add_lora,..."，多算子之间用英文逗号","分隔。
    
    说明：若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-math包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。

    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-math-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。
    
2. **安装自定义算子包**
   
    ```bash
    ./cann-ops-math-${vendor_name}_linux-${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/latest。注意自定义算子包不支持卸载。

### ops-math包

1. **编译ops-math包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：设置后表示不编译算子二进制文件，如需使用aclnn调用算子，该选项无需设置。
    - --soc：\$\{soc\_version\}表示NPU型号。鸿蒙Hardon PC / Hooper PC 使用"kirinx90"。


    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run" successfully created.
    ```

   \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-math包**

    ```bash
    ./cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

## 本地验证
ops-math包或者自定义算子包安装后，可以使用atc模型转换工具转换包含对应算子的onnx模型或者air模型。详细介绍参考[atc参数说明](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

模型转换成功后，需要调用鸿蒙应用开发接口，完成鸿蒙设备算子运行。详细介绍参考[鸿蒙应用开发](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cann-kit-guide)。



通过项目根目录build.sh脚本，可快速调用UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../context/build.md)。

- **执行算子UT**

	> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

    ```bash
  # 安装根目录下test相关requirements.txt依赖
  pip3 install -r tests/requirements.txt
  # 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
  bash build.sh -u --[opapi|ophost|opkernel] --ops=abs
  # 方式2: 编译并执行所有的UT测试用例
  # bash build.sh -u
  # 方式3: 编译所有的UT测试用例但不执行
  # bash build.sh -u --noexec
  # 方式4: 编译并执行对应功能的UT测试用例（选其一）
  # bash build.sh -u --[opapi|ophost|opkernel]
  # 方式5: 编译对应功能的UT测试用例但不执行（选其一）
  # bash build.sh -u --noexec --[opapi|ophost|opkernel]
    ```

    假设验证ophost功能是否正常，执行如下命令：
    ```bash
  bash build.sh -u --ophost
    ```

    执行完成后出现如下内容，表示执行成功。
    ```bash
  Global Environment TearDown
  [==========] ${n} tests from ${m} test suites ran. (${x} ms total)
  [  PASSED  ] ${n} tests.
  [100%] Built target math_op_host_ut
    ```
    \$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。