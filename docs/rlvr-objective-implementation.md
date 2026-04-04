# RLVR Objective Implementation

本文基于 `src/` 目录中的实际实现，按“在哪个文件写什么函数，该函数负责什么功能”的粒度，梳理当前仓库中 RLVR/GRPO 目标是如何落地的。

## 1. 训练入口

### `src/train/train_grpo.py`

#### `train()`

这是 RLVR 训练的总入口函数，负责把“参数解析、模型构建、数据集构建、奖励函数加载、Trainer 启动”串起来。

具体职责如下：

1. 通过 `HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))` 解析三类参数。
2. 校验配置合法性。
   - `DataArguments` 中不能同时设置 `nframes` 与 `fps`。
   - `lora_enable=True` 时要求 `freeze_llm=True`。
   - `vision_lora=True` 时要求 `freeze_vision_tower=True`。
3. 根据 `AutoConfig.from_pretrained(model_args.model_id)` 得到 `model_type`，加载对应的 Qwen VL 模型。
4. 在加载模型前调用不同的 monkey patch 函数，替换多模态 forward/vision 行为，使其兼容本仓库训练流程。
5. 调用 `configure_llm()`、`configure_vision_tower()`、`unfreeze_topk_layers()` 配置可训练参数范围。
6. 在 4bit/8bit 量化场景下，调用 `prepare_model_for_kbit_training()` 处理低比特训练。
7. 调用 `make_grpo_data_module()` 构建 GRPO 数据集。
8. 调用 `load_reward_funcs("src.train.reward_funcs")` 动态发现奖励函数。
9. 用 `QwenGRPOTrainer(...)` 创建 RLVR Trainer，并将 `reward_funcs` 传入。
10. 执行 `trainer.train()` 或断点续训。
11. 训练结束后保存模型、LoRA 权重和 processor。

这个函数是“RLVR objective 真正开始运行”的位置。任何要新增 RLVR 目标、改奖励、改 prompt 结构、改生成逻辑，最终都要在这里接入。

#### `configure_vision_tower(model, training_args, compute_dtype, device)`

作用是配置视觉塔的训练行为。

具体包括：

- 把 `model.visual` 移动到指定 dtype 和 device。
- 按 `freeze_vision_tower` 控制视觉 backbone 是否参与训练。
- 按 `freeze_merger` 控制 `visual.merger` 与 `deepstack_merger_list` 是否参与训练。

如果后续要做“只训 merger，不训视觉 backbone”的 RLVR 变体，应继续沿用这里的控制逻辑。

#### `configure_llm(model, training_args)`

作用是配置语言模型部分是否参与训练。

- 控制 `model.lm_head` 是否需要梯度。
- 控制 `model.language_model` 是否需要梯度。

如果要做“只训视觉侧 + 冻结语言侧”的 RLVR 实验，这个函数就是入口。

#### `unfreeze_topk_layers(model, k_llm=0, k_vis=0)`

作用是在整体冻结后，局部解冻 LLM 或视觉塔顶部若干层。

- `k_llm` 控制 `language_model.layers[-k_llm:]`
- `k_vis` 控制 `visual.blocks[-k_vis:]`

这不是 RLVR objective 本身的一部分，但它决定了 objective 作用到哪些参数。

#### `find_target_linear_names(model, ...)`

作用是收集 LoRA 注入目标模块名。

RLVR 文档里如果要说明 “LoRA 模式下训练哪些模块”，应明确引用这里的发现逻辑，而不是笼统写“自动训练所有线性层”。

#### `set_requires_grad(parameters, requires_grad)`

辅助函数，统一设置参数是否可训练。

## 2. 奖励函数注册与执行

### `src/utils.py`

#### `load_reward_funcs(module_path="train.reward_funcs", ...)`

这是奖励函数自动注册的核心函数。

它的实现方式不是“手工维护一个 reward 列表”，而是：

1. 用 `importlib.import_module(module_path)` 导入模块。
2. 用 `inspect.getmembers()` 找到模块中的可调用对象。
3. 只保留名称满足 `name.endswith("_reward")` 的函数。
4. 默认按源码出现顺序排序后返回。

因此，仓库当前的 RLVR 奖励扩展方式是：

- 在 `src/train/reward_funcs.py` 中新增一个函数；
- 函数名必须以 `_reward` 结尾；
- `train_grpo.py` 会自动加载并传给 `QwenGRPOTrainer`。

这意味着文档里不能写成“需要手动在训练脚本里注册奖励函数”，因为源码不是这么实现的。

### `src/train/reward_funcs.py`

#### `accuracy_reward(completions, assistant, **kwargs)`

这是当前主奖励函数，负责判断模型输出答案是否正确。

实现逻辑分两层：

1. 优先走数学/符号校验：
   - 对标准答案 `assistant` 调用 `parse(...)`。
   - 对模型输出 `completion` 再调用 `parse(...)`，并带上 `LatexExtractionConfig(...)` 与 `NormalizationConfig(...)`。
   - 调用 `verify(gold_parsed, answer_parsed)` 判断是否等价。
2. 如果标准答案无法被数学解析，则退化为字符串精确匹配：
   - `completion.strip().lower() == sol.strip().lower()`

输出是逐样本 reward 列表。

因此，当前仓库里的 RLVR 并不是“只做字符串匹配”，而是“优先做符号验证，失败时再退化成文本匹配”。

#### `format_reward(completions, **kwargs)`

这是格式奖励函数，负责约束输出格式是否匹配预设正则。

当前逻辑：

- 用正则 `r"^\\n.*?\\n\\n\\n.*?\\n$"` 检查 completion 格式。
- 匹配成功记 `1.0`，失败记 `0.0`。

它属于辅助奖励，而不是 correctness reward。

如果后续要继续扩展 RLVR objective，最合适的做法就是继续在这个文件新增类似：

- `grounding_reward(...)`
- `bbox_iou_reward(...)`
- `tool_call_reward(...)`

前提是函数名必须满足 `_reward` 后缀规则。

## 3. 数据是如何送进 RLVR 目标的

### `src/dataset/__init__.py`

#### `make_grpo_data_module`

这个符号从 `src/dataset/grpo_dataset.py` 导出，是 `train_grpo.py` 构造训练数据的固定入口。

文档里如果要给出“GRPO 数据准备入口”，应指向这里以及它实际导入的实现文件，而不是泛泛地说“Trainer 会自动构造数据”。

### `src/dataset/grpo_dataset.py`

#### `GRPODataset.__init__(...)`

负责初始化 RLVR 训练数据集对象。

主要职责：

- 读取 JSON 数据。
- 保存 `processor`、`data_args`、`model_id`。
- 解析图像/视频像素配置。
- 通过 `get_qwen_multimodal_settings(model_id)` 判断模型类型、patch size、视频 metadata 返回策略。
- 通过 `chat_template_uses_reasoning_prefill(...)` 判断是否支持 reasoning prefill。
- 在 `enable_reasoning=True` 但模型不支持时直接报错。

因此，`enable_reasoning` 不是一个对所有模型都通用的开关，而是受模型模板能力约束。

#### `GRPODataset.__getitem__(self, i)`

这是 RLVR 数据组织的关键函数。

它负责把单条样本整理成 Trainer 真正消费的字段。

具体过程：

1. 解析多模态输入
   - 若样本有 `image`，读取成 `images`
   - 若样本有 `video`，读取成 `videos`，并生成 `video_kwargs`
2. 将原始 `conversations` 转成 OpenAI 风格消息
   - 通过 `llava_to_openai(...)`
3. 取出用户消息和 assistant 标注
4. 根据是否存在 `reasoning` 字段，决定是否启用 reasoning prefill
5. 调用 `format_assistant_response(...)`，把 assistant 输出拆成：
   - `assistant_prefill`
   - `assistant_prompt`
6. 拼接 system prompt 与 user prompt，形成最终 `prompt`
7. 返回一个字典：
   - 必有：`prompt`, `assistant`
   - 可选：`images`, `videos`, `video_kwargs`

这里必须强调一个容易写错的点：

- `prompt` 是送给模型生成的前缀；
- `assistant` 是奖励函数中的参考答案；
- 当前数据集没有直接返回“完整 prompt+answer 拼接后的 token labels”给 RLVR；
- 它返回的是供 GRPO 生成与打分使用的原始结构化字段。

#### `make_grpo_data_module(model_id, processor, data_args)`

这是 `train_grpo.py` 实际调用的数据模块工厂函数。

当前实现很简单：

1. 构造 `GRPODataset(...)`
2. 返回 `dict(train_dataset=grpo_dataset, eval_dataset=None)`

因此，当前仓库里的 RLVR 训练默认没有单独接入 GRPO eval dataset。

## 4. RLVR 目标在 Trainer 中怎样执行

### `src/trainer/grpo_trainer.py`

#### `_identity_collator(features)`

作用是原样返回样本，不做默认 collate。

这样做的原因是 RLVR 训练需要保留原始字段：

- `prompt`
- `assistant`
- `images`
- `videos`
- `video_kwargs`

如果走默认 data collator，这些结构化字段很容易被提前处理掉，后续生成阶段就拿不到多模态原始输入。

#### `QwenGRPOTrainer.__init__(...)`

这是自定义 Trainer 初始化入口。

除了调用父类 `GRPOTrainer`，它还做了两件关键事：

1. 强制把 `self.data_collator` 改成 `_identity_collator`
2. 调用 `_ensure_mm_token_type_ids_generate_compat(...)`，修复部分模型在 `generate()` 时对 `mm_token_type_ids` 的兼容问题

因此，当前仓库中的 RLVR 不是直接裸用 `trl.GRPOTrainer`，而是做了一层面向 Qwen-VL 的多模态适配。

#### `_set_signature_columns_if_needed(self)`

这里覆写了 Trainer 的 signature columns，显式保留以下字段：

- `prompt`
- `assistant`
- `image`
- `images`
- `video`
- `videos`
- `video_kwargs`

作用是避免 `remove_unused_columns=True` 时，这些字段在进入训练步骤前被删除。

如果文档里写成“只保留 model forward 所需字段”，是不准确的；当前实现实际上是为训练步骤保留原始多模态字段。

#### `_generate_single_turn(self, prompts)`

这个函数负责“根据 prompt 真实生成 completion”。

它是 RLVR 中 rollout 的直接实现位置。

主要工作：

1. 从实例属性中取出当前 batch 的 `images`、`videos`、`video_kwargs`
2. 组装 `processor_kwargs`
   - 文本一定包含
   - 图像和视频按是否存在追加
3. 调用 `self.processing_class(**processor_kwargs)` 得到模型输入
4. 调用 `unwrapped_model.generate(...)` 生成 `prompt_completion_ids`
5. 从生成结果里切出 completion 部分
6. 截断到第一个 EOS
7. 返回：
   - `prompt_ids`
   - `completion_ids`
   - 额外的 `prompt_mm_token_type_ids`

这说明当前仓库的 RLVR objective 是“在线生成再打分”，不是“对离线 completion 直接计算 reward”。

#### `_generate_and_score_completions(self, inputs)`

这是整个 RLVR objective 最关键的函数。

它负责把数据集样本转成：

- prompt token
- completion token
- completion mask
- multimodal forward kwargs
- reward 计算所需的文本结果

建议把它理解为“rollout + 重新编码 + 打分前准备”的总调度函数。

它的实现职责包括：

1. 兼容不同输入格式
   - `list[dict]`
   - batched `dict`
2. 提取 `prompts`
3. 提取 `images` / `videos`
4. 把多模态输入暂存到 `self._current_images` / `self._current_videos`
5. 调用 `self._generate(prompts)` 执行生成
6. 清理暂存的多模态字段
7. 将生成得到的 `prompt_ids` / `completion_ids` pad 成 batch tensor
8. 构造 `attention_mask`
9. 为多模态 forward 重新调用 processor，得到额外的 `forward_kwargs`
10. 若存在 `mm_token_type_ids`，为 completion 段补零并拼接
11. 后续继续进入 TRL 的 logprob、advantage、loss 计算链路

从职责划分上说：

- “生成 completion”在 `_generate_single_turn`
- “把 completion 放回 RL objective 里做后续计算”在 `_generate_and_score_completions`

如果未来要加新的 RLVR 目标，比如对中间 step 单独奖励、对视觉 grounding token 单独 masking，这个函数是最需要改的地方。

#### `_ensure_mm_token_type_ids_generate_compat(model)`

作用是给某些被 monkey patch 或 liger 替换过的模型 forward 动态补一个 `mm_token_type_ids` 兼容层，避免 `generate()` 阶段因为 Python 签名里缺少该参数而报错。

这个函数不是奖励逻辑，但它是多模态 RLVR 能正常跑通的必要兼容层。

## 5. 参数定义在哪些文件

### `src/params.py`

#### `GRPOArguments`

这里定义了 RLVR/GRPO 训练的关键参数，包括：

- `beta`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `max_completion_length`
- `max_prompt_length`

因此，文档里如果描述 RLVR objective 的采样策略与 KL 约束来源，应明确对应这里的字段，而不是写成“这些参数写在训练脚本里”。

#### `DataArguments`

这里定义了数据相关参数，包括：

- `data_path`
- `eval_path`
- `image_folder`
- 图像/视频像素范围
- `fps`
- `nframes`
- `enable_reasoning`

RLVR 文档中所有和输入数据、视频采样、reasoning 开关有关的说明，都应落到这里。

#### `ModelArguments`

这里只保存模型 ID，本身不承载 RLVR 目标逻辑，但它决定后续 `train_grpo.py` 选择哪一种 Qwen VL 模型分支。

## 6. 当前仓库中“实现一个新的 RLVR objective”应该怎么做

如果目标是新增一个可训练使用的 RLVR objective，建议按下面的文件级别拆分实施：

### 方案 A：只新增奖励，不改 GRPO 主干

1. 在 `src/train/reward_funcs.py` 中新增函数
   - 例如：`grounding_reward(completions, assistant, images=None, **kwargs)`
   - 功能：对生成答案进行视觉 grounding 一致性打分
2. 保持函数名以 `_reward` 结尾
   - 这样 `load_reward_funcs()` 会自动加载
3. 不需要修改 `src/train/train_grpo.py`
   - 因为训练入口已经会自动导入奖励函数

这是当前仓库里扩展 RLVR objective 成本最低、最符合现有架构的做法。

### 方案 B：修改 rollout / 打分输入结构

如果仅靠 `reward_funcs.py` 不够，需要让 reward 看到更多信息，例如：

- 中间推理文本
- bbox
- 特定视觉 token
- 外部工具执行结果

则需要继续修改：

1. `src/dataset/grpo_dataset.py`
   - 在 `GRPODataset.__getitem__()` 中把额外字段放进返回字典
2. `src/trainer/grpo_trainer.py`
   - 在 `_set_signature_columns_if_needed()` 中保留新增字段
   - 在 `_generate_and_score_completions()` 中把新增字段传入 reward 计算链路
3. `src/train/reward_funcs.py`
   - 编写真正消费这些字段的奖励函数

### 方案 C：修改 RLVR 优化目标本身

如果不是“改 reward”，而是要改 GRPO loss 本身，例如：

- 改 advantage 归一化策略
- 改 KL 形式
- 加 auxiliary loss
- 对不同 token 区间做差异化加权

本次方案 C 需要额外明确一个约束：

- 原有固定 KL 系数 `beta` 路径必须保留；
- 新增 `use_beta_hat` 开关；
- 仅当 `use_beta_hat=True` 时，才在 `src/trainer/grpo_trainer.py` 内基于 `beta`、`T_cur`、`T_max` 计算 `beta_hat`；
- 当 `use_beta_hat=False` 时，loss 必须退化为原有固定 `beta` 逻辑；
- `alpha_hat` 的动态 advantage weighting 与 `use_beta_hat` 相互独立。

则主要修改文件应是：

1. `src/trainer/grpo_trainer.py`
   - 在 `QwenGRPOTrainer` 内覆写或扩展 TRL 的相关训练步骤
2. 必要时同步修改 `src/params.py`
   - 给新 objective 增加超参数
3. 仅当 reward 也变化时，再修改 `src/train/reward_funcs.py`

按本仓库当前结构，方案 C 的参数映射应写清楚：

- `beta` 继续对应 `src/params.py::GRPOArguments` 中原有 KL 系数配置；
- `use_beta_hat` 与 `t_max` 也应定义在 `src/params.py::GRPOArguments`；
- `src/trainer/grpo_trainer.py` 负责在运行时根据 `use_beta_hat` 决定使用固定 `beta` 还是动态 `beta_hat`；
- 日志层应区分“固定 KL 模式”和“动态 KL 模式”，避免文档或监控误以为所有训练都启用了余弦退火。

## 7. 文档落地建议

为避免文档继续和代码脱节，RLVR 相关文档建议固定遵守以下映射：

- “奖励函数在哪里加” 指向 `src/train/reward_funcs.py`
- “奖励函数如何被发现” 指向 `src/utils.py::load_reward_funcs`
- “GRPO 数据怎么组织” 指向 `src/dataset/grpo_dataset.py::GRPODataset.__getitem__`
- “多模态 rollout 在哪里做” 指向 `src/trainer/grpo_trainer.py::_generate_single_turn`
- “RLVR 训练入口在哪里” 指向 `src/train/train_grpo.py::train`
- “超参数定义在哪里” 指向 `src/params.py::GRPOArguments` 与 `src/params.py::DataArguments`

只有这样，后续修改代码时，文档才有明确对应关系，不容易再次写错。
