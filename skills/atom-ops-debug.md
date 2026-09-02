---
name: 实现ATOM算子工具
description: 为ATOM框架里的算子添加一层封装，添加运行时算子输入输出打印、dump和fallback等功能，并能根据dump file重新构建算子单测。
---

# 算子工具

为ATOM内所有算子添加封装，算子调用都统一使用atom.ops.xxx形式，在封装层实现算子各种辅助工具，并且能基于该工具实现大规模算子精度验证和性能统计等。


## 设计文档

请你按照以下步骤进行

```text
实现进度：
- [ ] 1) 算子基本信息统计
- [ ] 2) 算子工具功能需求
- [ ] 3) 工具方案设计
- [ ] 4) 插件注册已完成（如需要）
- [ ] 5) 冒烟测试通过
- [ ] 6) 功能测试
- [ ] 7) 算子工具使用文档
- [ ] 8) `recipes/atom_vllm` recipe 已新增/更新
- [ ] 9) CI 矩阵条目已添加
```

### 1) 算子基本信息统计

设计前先统计以下信息：

- ATOM中一共调用了哪些算子
- 这些算子的调用方式有哪几种
- 算子的参数和返回值有哪些类型

这是为了做设计方案时能兼容所有算子调用和参数处理。

### 2) 算子工具功能需求

应满足以下主体功能：

1. 需要有开关控制是否打开算子debug功能，且能根据开关等级对算子输入输出实现不同的功能

  - 打印算子名称，输入和输出
    * 等级1：对tensor类打印其shape，stride，dtype，device等信息，对int、float、string等内置类，直接打印数值，其它类型的参数打印请你列举出所有类型再做决定。
    * 等级2：打印tensor真实数据以及均值和方差，单个tensor默认打印不要超过20行，超出部分可只保留头尾。
    * 等级3：dump输入输出真实数据，且能根据该dump数据重建算子单测，你需要考虑较大tensor如何保存问题，如果多次运行则保存的数据会非常大。

  - 能在代码中随意设置标志位控制是否打印，以便用户可以选择打印不同的device和不同代码片段。不同device的dump数据应保存在不同的文件中。

2. 支持算子的fallback功能

  - 为每个算子实现naive小算子，添加开关决定哪些算子fall back回小算子。

3. 算子单测重建

  - 能够根据等级1的打印文件直接构建算子单测，根据等级3的dump数据构建真实数据算子单测。

### 3) 在 ATOM Model Runner 中注册

更新 `atom/model_engine/model_runner.py` 中的 `support_model_arch_dict`。

重要：遵循该文件的**当前本地风格**（例如，如果现有条目使用字符串 qualname 映射，就继续使用这种格式）。不要引入不同的映射格式。

### 4) 在插件模型注册表中注册

若支持插件模式，更新 `atom/plugin/register.py`：

- 为新的模型类添加 import
- 将架构 key 添加到 `_ATOM_SUPPORTED_MODELS`

如果模型仅用于 ATOM 原生路径，不通过插件路径使用，需要明确记录跳过插件注册的原因。

### 5) 冒烟测试

使用 `vllm serve` 运行快速冒烟测试：

```bash
vllm serve <model_path> \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

### 6) 精度验证

在另一个终端中运行精度验证：

```bash
lm_eval --model local-completions \
  --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

### 6.5) 将精度结果插入 Recipe

`lm_eval` 完成后，将实测结果插入对应的
`recipes/atom_vllm/<Model-Name>.md` 章节，表格格式如下：

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.93  |±  |0.0256|
|     |       |strict-match    |     5|exact_match|↑  |0.93  |±  |0.0256|
```

要求：

- 使用运行输出中的实际 `n-shot`、实测值和 stderr（不要过度四舍五入）。
- 将表格放在 recipe 中靠近 `lm_eval` 命令的位置。
- 包含原始结果 JSON 路径，便于追溯。

### 7) 添加 `recipes/atom_vllm` 条目

每个新增到 ATOM vLLM 插件支持范围内的模型，都需要在以下目录新增或更新使用 recipe：

- `recipes/atom_vllm/<Model-Name>.md`

recipe 至少应包含：

1. 一行范围说明：模型名 + 后端上下文（`ATOM vLLM plugin backend`）
2. `docker pull` 步骤（`rocm/atom-dev:vllm-latest`）
3. `vllm serve` 启动命令（包含精确模型路径，以及 TP、KV dtype、异步调度等关键参数）
4. 可选 benchmark 命令（`vllm bench serve` 或项目 benchmark 脚本）
5. 精度验证命令（`lm_eval --model local-completions ...`）
6. 使用标准 `|Tasks|Version|...|` 格式的精度结果表 + 原始 JSON 路径
7. 所有模型特定环境变量和注意事项（例如必要时的插件 attention 开关）

参考 `recipes/atom_vllm/` 中已有文件的风格（如 `DeepSeek-R1.md`、`Qwen3.5.md`、`Kimi-K2.5.md`）。

如果本次接入也更新了 CI 矩阵参数（TP size、环境变量、model id），确保 recipe 中的启动命令与这些 CI 设置保持一致。

### 8) 添加 CI 测试条目

更新 `.github/workflows/atom-vllm-oot-test.yaml`，并且只把模型条目添加到
**nightly accuracy 路径**。

不要将此模型添加到 pull-request OOT 矩阵（`jobs.atom-vllm-oot.strategy.matrix.include`）。

在 `jobs.prepare-oot-image -> step "Resolve image source and model matrix"` 中的
Python `models = [...]` 列表里添加：

```python
{
    "toggle_env": "RUN_NEW_MODEL_TP8",
    "model_name": "New-Model-Name TP8",
    "model_path": "org/model-name",
    "extra_args": "--tensor-parallel-size 8",
    "accuracy_test_threshold": 0.XX,
    "env_vars": "",
    "runner": "linux-atom-mi35x-8",
},
```

如果该模型应该是 **nightly only**（不能在手动 `workflow_dispatch` 中选择），
不要在 `on.workflow_dispatch.inputs` 下添加对应的 input toggle。

`accuracy_test_threshold` 必须基于真实评测结果设置，不要凭空猜测。

## 约束

- 不要修改带有 `@support_torch_compile` 装饰器的模型文件。
- 新环境变量只能定义在 `atom/utils/envs.py` 中。
- 量化行为需要与 HF `config.json` 中的 `quantization_config` 保持一致。
- 支持的量化格式包括 FP8、INT8、MXFP4、Quark。
- 接入过程中不要修改无关的模型映射。

## 输出格式

完成后报告：

1. 复用决策和理由
2. 修改的文件
3. 新注册的架构 key
4. 冒烟测试和精度结果
5. CI 条目详情
6. Recipe 文件路径和关键启动命令
7. 后续风险或 TODO

