### Floor 算子测试报告

**文档版本:** v1.0  
**测试日期:** 2025-11-13  
**测试环境:** CANN 8.5.0.alpha001  
**测试平台:** Ascend 910B2


### 测试方案

采用单算子调用的方式完成测试，测试流程包括：
```bash
1. 初始化 ACL 环境与设备。
2. 为每种数据类型（Float32 / Float16 / BFloat16）运行完整的测试：
  • 随机生成输入张量（shape = {1024, 1024, 4}）。
  • 计算 CPU 参考结果（基于 std::floor）。
  • 创建 NPU 张量并执行 aclnnFloor。
  • 从 NPU 拷贝结果并逐元素校验。
  • 多次迭代测试，统计平均执行时间、方差等性能指标（warmup次数不计入统计）。
3. 输出性能统计与验证结果。
4. 释放资源与结束。
```

### 测试步骤

1.安装固件，驱动，cann包
```bash
# 社区版默认安装目录位于：
  /usr/local/Ascend/
```

 

2.克隆算子包
```bash
git clone https://gitcode.com/cann/ops-math.git
```

3.编译算子
```bash
cd ops-math
bash build.sh --pkg --soc=ascend910b --ops=floor --vendor_name=custom --experimental
```

4.安装算子
```bash
./build_out/cann-ops-math-custom_linux-aarch64.run 
```

5.执行测试
```bash
cd ops-math
bash build.sh --run_example floor eager cust --vendor_name=custom --experimental
```

&nbsp;预期输出

```bash
[2025-11-13 16:58:29] This environment does not have the ASAN library, no need enable ASAN
[2025-11-13 16:58:29] CMAKE_ARGS:  -DENABLE_UT_EXEC=TRUE
[2025-11-13 16:58:29] ----------------------------------------------------------------
[2025-11-13 16:58:29] Start to run examples,name:floor mode:eager
[2025-11-13 16:58:29] Start compile and run examples file: ../experimental/math/floor/examples/test_aclnn_floor.cpp
[2025-11-13 16:58:29] pkg_mode:cust vendor_name:custom
[2025-11-13 16:59:01] 
[2025-11-13 16:59:01] ==================== Performance Summary (Float32) ====================
[2025-11-13 16:59:01] Correct count: 10 / 10
[2025-11-13 16:59:01] Total time: 2.264 ms
[2025-11-13 16:59:01] Min time: 0.122290 ms
[2025-11-13 16:59:01] Max time: 0.819401 ms
[2025-11-13 16:59:01] Avg time: 0.226444 ms
[2025-11-13 16:59:01] Std dev: 0.200373 ms
[2025-11-13 16:59:01] Variance: 0.040149 ms^2
[2025-11-13 16:59:01] 
[2025-11-13 16:59:01] ==================== Performance Summary (Float16) ====================
[2025-11-13 16:59:01] Correct count: 10 / 10
[2025-11-13 16:59:01] Total time: 2.113 ms
[2025-11-13 16:59:01] Min time: 0.188971 ms
[2025-11-13 16:59:01] Max time: 0.249140 ms
[2025-11-13 16:59:01] Avg time: 0.211254 ms
[2025-11-13 16:59:01] Std dev: 0.016993 ms
[2025-11-13 16:59:01] Variance: 0.000289 ms^2
[2025-11-13 16:59:01] 
[2025-11-13 16:59:01] ==================== Performance Summary (BFloat16) ====================
[2025-11-13 16:59:01] Correct count: 10 / 10
[2025-11-13 16:59:01] Total time: 1.397 ms
[2025-11-13 16:59:01] Min time: 0.113890 ms
[2025-11-13 16:59:01] Max time: 0.189480 ms
[2025-11-13 16:59:01] Avg time: 0.139740 ms
[2025-11-13 16:59:01] Std dev: 0.026211 ms
[2025-11-13 16:59:01] Variance: 0.000687 ms^2
[2025-11-13 16:59:01] 
[2025-11-13 16:59:01] ========== All Tests Passed ==========
```


6.卸载刚才安装的算子
```bash
rm -rf /usr/local/Ascend/latest/opp/vendors/*
```


7.再次执行测试
```bash
bash build.sh --run_example floor eager --experimental
```

### 测试结果

#### 精度

| 测试项 | AscendC | TBE |
|---------|----------|-----|
| 正确性 | 100/100 | 100/100 |


---

#### 性能

| 数据类型 | 平均耗时 (AscendC) | 平均耗时 (TBE) | 差异 | 是否达标 |
|-----------|--------------------|----------------|------|------------------|
| Float32   | 0.1872 ms | 0.1913 ms | **-2.1%** | ✅ 是 |
| Float16   | 0.1979 ms | 0.2008 ms | **-1.4%** | ✅ 是 |
| BFloat16  | 0.1977 ms | 0.1953 ms | **+1.2%** | ✅ 是 |

---