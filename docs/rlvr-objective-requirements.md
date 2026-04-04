# RLVR 目标函数改造需求文档

## 1. 背景

当前项目在 RLVR/GRPO 训练阶段使用现有目标函数，其中包含 policy optimization 项和 KL penalty 项。为提升训练稳定性、增强中等难度样本的利用效率，并在需要时使 KL 约束随训练进度动态衰减，需要将目标函数升级为带动态 advantage weighting，并支持可选 cosine-annealed KL penalty 的版本。

本次改造保留原有 KL 系数配置 `beta` 作为默认行为。仅当新增开关 `use_beta_hat=True` 时，才以它作为基础系数进一步计算训练时实际使用的动态 KL 系数 `beta_hat`；否则训练时继续使用固定 KL 系数 `beta`。

## 2. 需求目标

将项目中的 RLVR/GRPO 目标函数改造为如下形式：

$$
J_{GRPO}=\mathbb{E}_{q,\{o_i\}\sim\pi_{\theta_{old}}}\left[
\frac{\hat{\alpha}_{\{o_i\}}}{G}
\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{l=1}^{|o_i|}
\min\left(r_{i,l}(\theta)\hat{A}_{i,l},\operatorname{clip}(r_{i,l}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{i,l}\right)
-\hat{\beta}D_{KL}(\pi_\theta\|\pi_{ref})
\right]
$$

其中新增一个动态参数 `alpha_hat`，并新增一个可选启用的动态 KL 系数 `beta_hat`。

## 3. 参数定义

### 3.1 动态 advantage 权重 `alpha_hat`

$$
\hat{\alpha}_{\{o_i\}_{i=1}^{G}}=\sigma \cdot \mathrm{mean}(\{R_i^{acc}\}_{i=1}^{G}) \cdot \left(1-\mathrm{mean}(\{R_i^{acc}\}_{i=1}^{G})\right)
$$

参数说明：

- `R_i^acc` 在本项目中对应现有字段 `accuracy_reward`
- `mean({R_i^acc})` 表示当前 group 内 `accuracy_reward` 的均值
- `sigma` 为缩放系数，固定为 `7.2`

语义说明：

- 当 group 的平均 `accuracy_reward` 接近 `0.5` 时，`alpha_hat` 较大，表示中等难度样本具有更高训练权重
- 当平均 `accuracy_reward` 接近 `0` 或 `1` 时，`alpha_hat` 接近较小值，表示过难或过易样本的训练权重降低

### 3.2 可选动态 KL 系数 `beta_hat`

$$
\hat{\beta}=\frac{\beta}{2}\left(1+\cos\left(\pi\frac{T_{cur}}{T_{max}}\right)\right)
$$

参数说明：

- `beta` 为原有 KL 系数配置值，并继续作为基础 KL 系数保留
- `T_cur` 为当前回答在本次训练中的 training steps
- `T_max` 为可配置的超参数，用于控制 KL 系数余弦退火周期
- `use_beta_hat` 为布尔开关，控制是否启用动态 KL 系数

语义说明：

- 当 `use_beta_hat=False` 时，训练时使用固定 KL 系数 `beta`
- 当 `use_beta_hat=True` 时，训练初期，`beta_hat` 接近 `beta`
- 当 `use_beta_hat=True` 时，随着当前回答对应的 training steps 增长，`beta_hat` 按余弦曲线逐步衰减
- 当 `use_beta_hat=True` 且 `T_cur` 接近配置的 `T_max` 时，`beta_hat` 下降到接近 `0`

## 4. 功能需求

### 4.1 目标函数替换

系统必须将当前 RLVR/GRPO 的目标函数替换为新公式，并满足以下要求：

- policy 主项前增加 `alpha_hat`
- 原有 KL 系数继续保留，并作为 `beta_hat` 的基础输入
- 当 `use_beta_hat=True` 时，KL penalty 项使用 `beta_hat`
- 当 `use_beta_hat=False` 时，KL penalty 项继续使用固定 `beta`
- 保持现有 clipped objective 的核心计算形式不变

### 4.2 参数来源要求

系统必须保证损失计算阶段能够直接获取以下输入：

- `accuracy_reward`
- `beta`
- `T_cur`
- `T_max`
- `sigma`
- `use_beta_hat`

其中：

- `accuracy_reward` 作为 `alpha_hat` 的唯一 reward 来源
- `T_cur` 必须来自当前回答在本次训练中的实际 training steps 计数
- `T_max` 必须作为可配置超参数提供，而不是从全局训练总步数隐式推导
- `beta` 来自当前项目原有 KL 系数配置
- `use_beta_hat` 控制损失计算阶段使用 `beta_hat` 还是固定 `beta`

### 4.3 数据流要求

系统必须保证以下数据流成立：

1. `accuracy_reward` 可从 reward 计算链路传递到 group 级损失计算链路
2. 当前回答对应的 `T_cur`、配置项 `T_max` 与开关 `use_beta_hat` 可传递到损失函数或其上游调用逻辑
3. 损失函数在单次前向计算中可以同时获取 `accuracy_reward`、policy ratio、advantage、KL divergence，以及 KL 系数选择上下文

### 4.4 配置需求

需要新增或明确以下配置项：

```yaml
rlvr:
  objective:
    advantage_weighting:
      sigma: 7.2
      reward_key: accuracy_reward
    kl_penalty:
      beta: <原KL配置值>
      use_beta_hat: false
      t_max: <可配置超参数>
      schedule: cosine
```

要求如下：

- `reward_key` 固定使用 `accuracy_reward`
- `sigma` 默认值为 `7.2`
- `beta` 沿用现有 KL 参数
- `use_beta_hat` 默认为 `false`
- `t_max` 在 `use_beta_hat=True` 时必须为正数，且作为超参数显式配置
- `schedule=cosine` 仅在 `use_beta_hat=True` 时生效

### 4.5 日志与可观测性需求

训练过程中至少需要记录以下指标：

- `alpha_hat`
- `beta_hat`
- `mean_accuracy_reward`
- `beta`
- `use_beta_hat`
- `T_cur`
- `T_max`
- `policy_loss`
- `kl_loss`
- `total_loss`

目的如下：

- 验证 `alpha_hat` 是否按 `accuracy_reward` 均值变化
- 验证 `use_beta_hat=True` 时 `beta_hat` 是否按训练步数执行余弦退火
- 验证 `use_beta_hat=False` 时 KL 项是否继续使用固定 `beta`
- 验证总损失在新目标函数下是否按预期响应

## 5. 具体修改部位

以下为必须改动的实现部位。由于本文件面向需求定义，描述采用模块职责级别。

### 5.1 RLVR/GRPO 损失函数计算模块

需要修改：

- 目标函数主入口
- policy 主项聚合位置
- KL penalty 组装位置

改动内容：

- 在原有 clipped objective 外层乘以 `alpha_hat`
- 保留原 KL 系数配置，并在 `use_beta_hat=True` 时于损失计算时基于它得到 `beta_hat`
- 在 `use_beta_hat=False` 时保持原固定 KL 系数逻辑
- 保持 KL divergence 本身的计算定义不变

### 5.2 Reward 数据整理模块

需要修改：

- `accuracy_reward` 生成后的封装与传递逻辑

改动内容：

- 明确 `R_i^acc` 与 `accuracy_reward` 在语义上都表示回答是否正确
- 确保其可在 group 级别参与 `mean_accuracy_reward` 计算
- 若当前 reward 容器中存在多个 reward 项，仅抽取 `accuracy_reward` 供 `alpha_hat` 使用
- 是否在代码中直接以 `accuracy_reward` 替换 `R_i^acc`，必须以实际字段来源、数据结构和调用链审查结果为准

### 5.3 Trainer 状态传递模块

需要修改：

- 当前回答 training steps 与 `T_max` 超参数的注入链路

改动内容：

- 将 `T_cur` 与 `T_max` 传给损失计算上下文
- 将 `use_beta_hat` 传给损失计算上下文
- 保证当前回答的 training steps 计数方式与 KL 退火语义一致

### 5.4 配置定义模块

需要修改：

- RLVR/GRPO 配置结构
- 默认值定义

改动内容：

- 增加 `sigma`
- 保留原 KL 配置项，并明确其作为动态 KL 的基础系数 `beta`
- 增加 `use_beta_hat`
- 明确 `schedule=cosine` 仅在启用动态 KL 时生效

### 5.5 日志模块

需要修改：

- 训练日志采集项
- 指标上报项

改动内容：

- 增加 `alpha_hat`、`mean_accuracy_reward`、`use_beta_hat` 等监控字段
- 在 `use_beta_hat=True` 时增加 `beta_hat` 相关监控字段
- 确保调试和回归分析时能定位新目标函数的行为变化

## 6. 验收标准

需求完成后必须满足以下条件：

1. 训练损失已使用新目标函数
2. `alpha_hat` 由 `accuracy_reward` 的 group 均值计算得到
3. 当 `use_beta_hat=True` 时，`beta_hat` 由保留的原 KL 系数 `beta`、当前回答的 `T_cur` 以及配置超参数 `T_max` 按余弦公式计算得到
4. 当 `use_beta_hat=True` 时，KL 项在训练时使用 `beta_hat`，且 `beta_hat` 的基础输入来自原 KL 系数；当 `use_beta_hat=False` 时，KL 项继续使用固定 `beta`
5. 日志中可观测到 `alpha_hat` 的变化轨迹，并在 `use_beta_hat=True` 时可观测到 `beta_hat` 的变化轨迹
6. 训练流程可以正常启动、训练、保存和恢复

## 7. 风险与注意事项

- 若 `accuracy_reward` 当前没有直接传到 loss 层，需要补齐数据链路
- 若当前回答的 training steps 仅在局部流程中维护，但未暴露给算法模块，需要增加参数透传
- 若现有系统已经存在其他 KL schedule，需要停用旧 schedule，避免重复调度
- 若 `use_beta_hat=False` 时仍意外进入动态 KL 调度，会改变现有训练语义，因此必须显式保证退化路径正确
- 若 `accuracy_reward` 理论取值不在 `[0, 1]`，需要先确认公式适配性或增加边界保护

## 8. 结论

本次改造的本质是：

- 用 `accuracy_reward` 驱动样本级训练权重分配
- 在保留原 KL 系数配置的前提下，通过 `use_beta_hat` 控制是否基于当前回答的 training steps 和可配置超参数 `T_max` 生成训练时使用的动态 KL 惩罚强度
- 在不改变现有 KL divergence 定义的前提下，升级 RLVR 阶段的优化目标

实现应聚焦于损失函数、reward 透传、trainer 步数注入、配置与日志五个位置。
