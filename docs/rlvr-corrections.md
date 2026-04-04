# RLVR 文档修订说明

本文根据 `src/` 中的实际实现，对 RLVR 文档里容易写错或已经与代码不一致的表述做统一纠正。

## 1. 奖励函数不是手动注册，而是自动发现

错误表述：

`train_grpo.py` 里手动维护 reward 函数列表。

订正：

当前仓库通过 `src/utils.py` 中的 `load_reward_funcs()` 自动导入 `src.train.reward_funcs` 模块，并自动收集所有名称以 `_reward` 结尾的可调用对象。

这意味着：

- 新增奖励函数时，默认不需要改训练入口；
- 只要把函数写进 `src/train/reward_funcs.py` 且命名满足规则即可。

## 2. RLVR 的 correctness reward 不只是字符串匹配

错误表述：

当前仓库仅通过字符串完全一致判断回答是否正确。

订正：

`src/train/reward_funcs.py` 中的 `accuracy_reward()` 会优先尝试：

1. 解析标准答案；
2. 解析模型输出；
3. 用符号验证工具 `verify(...)` 判断两者是否等价。

只有在标准答案无法被数学解析时，才退化为大小写无关的字符串精确匹配。

因此，文档应写成：

当前 correctness reward 是“符号验证优先，字符串匹配兜底”。

## 3. `assistant` 不是完整训练标签，而是奖励参考答案

错误表述：

GRPO 数据集直接返回用于监督训练的完整标签序列。

订正：

`src/dataset/grpo_dataset.py` 中 `GRPODataset.__getitem__()` 返回的核心字段是：

- `prompt`
- `assistant`
- `images` / `videos`
- `video_kwargs`

其中：

- `prompt` 是生成前缀；
- `assistant` 是奖励函数要对比的参考答案；
- 当前实现并不是 SFT 那种“构造完整 labels 张量再算交叉熵”。

因此 RLVR 文档不应混淆 SFT 与 GRPO 的数据组织方式。

## 4. 当前仓库不是直接裸用 `trl.GRPOTrainer`

错误表述：

本项目完全依赖原生 `trl.GRPOTrainer`，没有额外适配。

订正：

当前仓库定义了 `src/trainer/grpo_trainer.py::QwenGRPOTrainer`，至少做了以下适配：

- 使用 `_identity_collator` 保留原始多模态字段；
- 覆写 `_set_signature_columns_if_needed()` 保留 `prompt`、`assistant`、`images`、`videos` 等字段；
- 覆写多模态生成逻辑；
- 通过 `_ensure_mm_token_type_ids_generate_compat()` 修复部分模型生成兼容性问题。

因此文档应明确写成：

当前 RLVR 实现是在 TRL 的 GRPOTrainer 基础上，增加了一层 Qwen-VL 多模态适配。

## 5. rollout 是在线生成，不是对离线 completion 直接打分

错误表述：

数据集里已经提供 completion，训练时只做离线 reward 计算。

订正：

`src/trainer/grpo_trainer.py` 中：

- `_generate_single_turn()` 负责调用 `generate()` 在线生成 completion；
- `_generate_and_score_completions()` 再把生成结果转回 token/tensor 形式，供后续 logprob、reward、advantage、loss 链路使用。

因此，当前实现属于在线 rollout 的 RLVR，而不是纯离线打分。

## 6. `enable_reasoning` 不是对所有模型都可用

错误表述：

只要把 `--enable_reasoning True` 打开，任何模型都能使用 reasoning 字段。

订正：

`src/dataset/grpo_dataset.py` 会先判断模型是否支持官方 reasoning chat template。

只有受支持的模型才允许启用 reasoning prefill；否则会直接报错。

此外：

- 对某些模型，reasoning 是必填；
- 对某些模型，允许 reasoning 可选；
- 这不是统一行为。

因此文档里必须把“是否支持 reasoning”写成模型能力约束，而不是通用配置项。

## 7. `beta`、采样参数和长度参数定义在参数类里，不在文档示例里“临时生效”

错误表述：

`beta`、`temperature`、`top_p`、`max_completion_length` 等参数只是脚本示例中的可选项。

订正：

这些字段在 `src/params.py::GRPOArguments` 中有正式定义，属于训练配置的一部分。

因此文档应把它们描述为：

- RLVR/GRPO 训练超参数；
- 由参数类统一声明；
- 训练入口 `src/train/train_grpo.py` 负责解析并传入 Trainer。

## 8. RLVR 扩展点应该按文件职责描述，不能只写“修改 reward_funcs.py”

错误表述：

如果要实现新的 RLVR objective，只需要改 `reward_funcs.py`。

订正：

分三种情况：

1. 只改奖励函数
   - 改 `src/train/reward_funcs.py` 即可。
2. 要让奖励看到更多字段
   - 同时改 `src/dataset/grpo_dataset.py` 和 `src/trainer/grpo_trainer.py`。
3. 要改 GRPO loss / advantage / KL 形式
   - 主要改 `src/trainer/grpo_trainer.py`，必要时改 `src/params.py`。

所以更准确的文档写法应是：

“RLVR objective 的扩展点分为数据组织、rollout 生成、奖励函数、优化目标四层，分别对应不同文件。”

## 9. 动态 KL 不是默认强制启用，而应由显式开关控制

错误表述：

只要文档里引入 `beta_hat`，训练时就统一改为使用动态 KL 系数。

订正：

本次方案 C 需要保留原有固定 KL 系数 `beta` 的训练语义。

更准确的需求表述应是：

- 新增 `use_beta_hat` 作为显式布尔开关；
- 当 `use_beta_hat=True` 时，训练时使用基于 `beta`、`T_cur`、`T_max` 计算得到的 `beta_hat`；
- 当 `use_beta_hat=False` 时，训练时继续使用原有固定 KL 系数 `beta`；
- `alpha_hat` 的动态 advantage weighting 与是否启用 `beta_hat` 相互独立。

因此，文档不能写成“KL penalty 项统一使用 `beta_hat`”，而应写成“KL penalty 项在启用动态 KL 时使用 `beta_hat`，否则退化为原固定 `beta`”。
