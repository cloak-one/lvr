# Document B - Macro Structure and Workflow

## 目录组织（仅 src）

### 顶层职责
- `src/model`: 模型结构扩展（LVR 主模型与 head）
- `src/dataset`: SFT/LVR/DPO/GRPO 数据集与样本组织
- `src/train`: 各训练阶段入口、forward monkey patch、reward/loss 辅助
- `src/trainer`: 不同训练范式的 Trainer 实现（SFT/LVR/DPO/GRPO）
- `src/constants.py`: 全局常量（如 `IGNORE_INDEX`）
- `src/params.py` / `src/params_vanilla.py`: 参数 dataclass（模型、数据、训练）
- `src/train/train_utils.py`: 保存权重、LoRA 状态等通用训练工具
- `src/s3_checkpoints_lvr.py`: 在线 checkpoint 同步（可选）

## Pipeline 视图

### 1) SFT Pipeline
- Start entry（src 内）: `src/train/train_sft.py::train`
- Main modules:
  - `src/train/monkey_patch_forward.py`
  - `src/dataset/sft_dataset.py`
  - `src/trainer/sft_trainer.py`
- Inputs:
  - `ModelArguments/DataArguments/TrainingArguments`
  - 图文或视频对话样本
- Outputs:
  - SFT checkpoint
- Handoff:
  - 可作为后续 LVR 或 DPO 的初始化模型

### 2) LVR Stage-1 Pipeline
- Start entry（src 内）: `src/train/train_lvr.py::train`
- Main modules:
  - `src/model/qwen_lvr_model.py`
  - `src/train/monkey_patch_forward_lvr.py`
  - `src/dataset/lvr_sft_dataset.py` 或 `src/dataset/lvr_sft_dataset_packed.py`
  - `src/trainer/lvr_trainer.py`
- Inputs:
  - LVR 参数（`lvr_head`, `latent_end_token`, `loss_lvr_fct`, `loss_lvr_lambda`）
  - 含图像与推理标注信息的数据
- Outputs:
  - 带 LVR 能力的 stage1 checkpoint
- Handoff:
  - 交给 GRPO stage2（`train_grpo.py`）继续强化

### 3) DPO Pipeline
- Start entry（src 内）: `src/train/train_dpo.py::train`
- Main modules:
  - `src/dataset/dpo_dataset.py`
  - `src/trainer/dpo_trainer.py`
  - `src/train/monkey_patch_forward.py`
- Inputs:
  - prompt/chosen/rejected 偏好样本
  - model 与（可选）ref_model
- Outputs:
  - DPO 优化后的权重
- Handoff:
  - 输出可直接用于推理或继续实验

### 4) GRPO Stage-2 Pipeline
- Start entry（src 内）: `src/train/train_grpo.py::train`
- Main modules:
  - `src/model/qwen_lvr_model.py`
  - `src/train/monkey_patch_forward_lvr_rl.py`
  - `src/dataset/grpo_dataset.py`
  - `src/train/reward_funcs.py`
  - `src/trainer/grpo_trainer.py`
- Inputs:
  - stage1 模型权重
  - prompt/assistant 数据
  - reward functions + 权重
- Outputs:
  - GRPO 优化后的 stage2 权重
- Handoff:
  - 最终推理模型

## 端到端数据流（src 证据）

### Flow-A: 参数与模型配置
1. `src/train/train_*.py` 使用 `HfArgumentParser` 解析参数 dataclass。
2. `src/params.py` 或 `src/params_vanilla.py` 承载训练开关与超参。
3. `train_lvr.py` / `train_grpo.py` 中注入 monkey patch。
4. 对 LVR 路径，`QwenWithLVR` 初始化 `lvr_head`、可选 latent end embedding。

### Flow-B: 数据构造与 batch 形态
1. `src/dataset/*_dataset.py` 把原始样本转成模型输入。
2. LVR 数据在 `SupervisedDatasetLVR` 中处理 bbox 与 token 索引。
3. GRPO 数据在 `GRPODataset.__getitem__` 形成 `prompt` + `assistant`。
4. 部分路径使用 packed dataset 提升 token 利用率。

### Flow-C: 前向、loss/reward、优化
1. SFT: 标准 CE。
2. LVR: `CE + LVR loss (+ mode switch loss 可选)`，核心在 `monkey_patch_forward_lvr.py`。
3. DPO: chosen/rejected 对比目标，核心在 `dpo_trainer.py`。
4. GRPO: 生成 completion -> reward 计算 -> 结合 ref model 做策略更新，核心在 `grpo_trainer.py`。
5. `lvr_trainer.py` 对视觉塔/merger/lvr_head 提供分层学习率。

### Flow-D: 保存与恢复
1. `src/train/train_utils.py::safe_save_model_for_hf_trainer` 负责保存。
2. 训练入口按是否有 checkpoint 决定恢复逻辑。
3. 可选地通过 `src/s3_checkpoints_lvr.py` 做在线 checkpoint 同步。

## SFT / LVR / DPO / GRPO 分歧矩阵

| 维度 | SFT | LVR Stage-1 | DPO | GRPO Stage-2 |
|---|---|---|---|---|
| 入口 | `train_sft.py` | `train_lvr.py` | `train_dpo.py` | `train_grpo.py` |
| 模型 | Qwen2.5-VL | QwenWithLVR | Qwen2.5-VL | QwenWithLVR |
| Patch | `monkey_patch_forward.py` | `monkey_patch_forward_lvr.py` | `monkey_patch_forward.py` | `monkey_patch_forward_lvr_rl.py` |
| 数据 | `sft_dataset.py` | `lvr_sft_dataset*.py` | `dpo_dataset.py` | `grpo_dataset.py` |
| 目标 | CE | CE + LVR | DPO objective | GRPO/PPO-like objective |
| 奖励函数 | 无 | 无 | 无 | `reward_funcs.py` |
| ref model | 无 | 无 | 可选 | 必需（训练逻辑依赖） |

## 关键扩展点（从 src 视角）
- 改模型头: `src/model/lvr_heads.py` + `src/model/qwen_lvr_model.py::_init_lvr_head`
- 改 LVR loss: `src/train/monkey_patch_forward_lvr.py::set_lvr_loss_fct`
- 改 reward: `src/train/reward_funcs.py`（新增 `_reward` 函数）
- 改 dataset: `src/dataset/lvr_sft_dataset.py` 或 `src/dataset/grpo_dataset.py`
- 改 trainer 策略: `src/trainer/lvr_trainer.py` / `src/trainer/grpo_trainer.py`

## Open Questions and Validation Gaps
- `src` 中未直接包含脚本层面的完整实验编排参数，跨实验对比需外部配置补充（out-of-scope）。
- `GRPOTrainer` 大量逻辑在 `src/trainer/grpo_trainer.py` 内，本文聚焦关键流，不覆盖每个分支的数值稳定性细节。
- 当前项目中评估集通常为 `None`，离线评估协议与指标定义在 `src` 中并不完整，需要单独补齐。
