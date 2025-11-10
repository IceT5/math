
> 说明：针对基于[Ascend/samples](https://gitee.com/ascend/samples/tree/master)仓贡献的算子，请参考[算子工程迁移](#算子工程迁移)完成存量算子往本项目工程迁移的操作。

# 算子工程迁移

根据算子开发流程介绍，将待迁移算子内容迁移至对应交付件。适配差异点：

**1. ${op_name}_def.cpp**

将待迁移算子${op_name}.cpp中算子信息库相关内容迁移至该文件后，需要去掉SetInferShape和SetTiling内容。

**2. ${op_name}_tiling.cpp**

将待迁移算子${op_name}.cpp中Tiling实现相关内容迁移至该文件后，在该文件中完成Tiling函数注册。

```CPP
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc);
```

**3. ${op_name}_infershape.cpp**

图模式场景需要适配该文件，将待迁移算子${op_name}.cpp中InferShape实现相关内容迁移至该文件后，在该文件中完成InferShape函数注册。

```CPP
IMPL_OP_INFERSHAPE(${op_name}).InferShape(InferShape);
```

**4. ${op_name}_graph_infer.cpp**

图模式场景需要适配该文件，将待迁移算子${op_name}.cpp中InferDataType实现相关内容迁移至该文件后，在该文件中完成InferDataType函数注册。

```CPP
IMPL_OP(${op_name}).InferDataType(InferDataType);
```