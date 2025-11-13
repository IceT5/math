### FloorDiv (AscendC / AiCore) —— 完整实现说明

本目录为 **FloorDiv** 的 NPU（AiCore, AscendC）实现，接口与工程中 `angle_v2` / `add_lora` 等算子保持一致。
已提供：**AscendC 内核**、**Host 侧 Tiling**、**opc 二进制配置**、**AICPU 兜底**、**UT 测试** 对接。

---

#### 1. 功能与语义

* **算子名**：`FloorDiv`（逐元素除法并向下取整）。
* **输入/输出**：`out = floor(x / y)`；默认 **等形状**（no-broadcast）在 AiCore 上直接计算；若存在广播，由 Host 侧接口判定并可回落 AICPU（或后续扩展 kernel 增加广播索引）。
* **整数语义**：数学 floor 除法：先趋零截断 `q = x / y`，余数 `r = x - q*y`，若 `r != 0` 且 `sign(r) != sign(y)` 则 `q--`。
* **浮点语义**：`out = floor(x / y)`。

**支持数据类型**

* 浮点：`float16`、`float32`、`bf16`
* 整数：`int32`、`int16`、`int8`、`uint8`

> 可扩展到 `int64/double`：按相同模式补 dtypeKey、模板与 `binary.json` 条目。

---

#### 2. 代码结构

```
math/floor_div/
├─ op_kernel/
│  ├─ floor_div_core.h   # 纯 C++ 核心数学（AICPU/AiCore 均可复用）
│  ├─ floor_div.h        # AscendC 内核（UB tile、GM↔UB 拷贝、UB 内计算）
│  └─ floor_div.cpp      # AscendC kernel 入口（按 dtypeKey 派发）
├─ op_host/
│  ├─ floor_div_tiling.h/.cpp   # Tiling 定义与计算；注册 FloorDivTilingData
│  └─ config/
│     ├─ ascend910b/
│     │  ├─ floor_div_binary.json          # opc 二进制映射（占位 bin 名）
│     │  └─ floor_div_simplified_key.ini   # 简化 key：default=0
│     └─ ascend910_93/
│        ├─ floor_div_binary.json
│        └─ floor_div_simplified_key.ini
└─ tests/
   └─ ut/op_host/op_api/test_aclnn_floor_divide.cpp  # UT（已有）
```

---

#### 3. 启用 AiCore 路径

* 在 `math/floor_div/op_host/op_api/floordiv.cpp` 中确保：

  * `IsAiCoreSupport(...)` 对以下 dtype 返回 `true`：`float16/float32/bf16/int32/int16/int8/uint8`；
  * 如需稳妥处理广播，**仅等形状走 AiCore**，其它回落 AICPU；
* 满足条件时自动通过 `ADD_TO_LAUNCHER_LIST_AICORE(FloorDiv, ...)` 启动 AscendC 内核，否则回落 AICPU。

---

#### 4. 构建与二进制（opc）

1. **二进制映射**：`op_host/config/**/floor_div_binary.json` 已为各 dtype 预置条目：
   `op_type:"FloorDiv"`, `kernel_name:"floor_div"`，`bin_filename` 为占位名。
2. **编译 bin**：按工程既有流程（参考 `angle_v2`）执行 `opc` 生成 bin，
   然后把**实际生成的 bin 文件名**回填到 `bin_filename`。
3. **CMake 构建**：工程已有全局规则会收集 `op_kernel/*.cpp` 与 `op_host/*.cpp`，直接按仓库既有脚本编译。

> 本内核采用 ABI：`floor_div(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)`，与常规启动宏兼容。

---

#### 5. 单元测试 / 自验证

* 已有 UT：`tests/ut/op_host/op_api/test_aclnn_floor_divide.cpp`
* 编译后运行 `ctest`（或工程既有测试脚本）。
* 预期：

  * **功能**：整型/浮点/负数场景正确；（若 UT 含广播）应自动回落 AICPU 保证正确。
  * **性能**：等形状大 tensor 在 AiCore 上优于/接近 TBE 参考（可视向量化与 tile 调优）。

---

#### 6. 实现与调优要点

* **扁平化 + 均匀分核**：`lenPerCore = ceil(totalLength/coreNum)`，尾部收敛。
* **UB tile**：默认 `TILE = 1024`；可按 910B UB 容量放大或加 ping-pong。
* **向量化**：当前使用标量循环保证语义可读与正确；后续可替换为向量指令以提升吞吐。
* **广播处理**：当前建议在 `IsAiCoreSupport` 阶段回落；如需 AiCore 支持广播，可扩展 tiling 传入 strides/shape，并在 kernel 做索引映射。

---

#### 7. 变更记录

* 新增：`op_kernel/*`、`op_host/floor_div_tiling.*`、`config/**/floor_div_*.{json,ini}`
* 调整：无（不改动现有 AICPU / API 接口；仅需将 `IsAiCoreSupport` 打开）
