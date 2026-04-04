# Document A - Core Micro View

## 目标
本文件从 `src/**` 中抽取最小核心定义，帮助快速串联 LVR 项目的关键训练闭环。

## 1) 最小模型定义

### A. `QwenWithLVR`
- 文件: `src/model/qwen_lvr_model.py`
- 符号: `QwenWithLVR`
- 核心性: 在 Qwen2.5-VL 基础上注入 LVR head 与 latent end token 能力，是 LVR/GRPO 阶段模型基座。
- 关键依赖: `LVRHead`, `LVRHeadGLU`, `Qwen2_5_VLForConditionalGeneration`

```python
class QwenWithLVR(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.lvr_head:
            self._init_lvr_head(config.lvr_head_type)
        if config.latent_end_token:
            self._init_lvr_latent_end_emb()

    def _init_lvr_head(self,lvr_head_type):
        print(f"Detected LVR Head Type: '{lvr_head_type}'")
        if lvr_head_type == 'simple':
            self.lvr_head = LVRHead(hidden_size=self.config.hidden_size)
        elif lvr_head_type == 'glu':
            self.lvr_head = LVRHeadGLU(hidden_size=self.config.hidden_size,
                                       intermediate_size=self.config.intermediate_size,
                                       hidden_act=self.config.hidden_act)
```

### B. `LVRHead` / `LVRHeadGLU`
- 文件: `src/model/lvr_heads.py`
- 符号: `LVRHead`, `LVRHeadGLU`
- 核心性: 定义隐式推理表示的映射头，直接决定 LVR 对齐能力。
- 关键依赖: `nn.Linear`, `LayerNorm`, `ACT2FN`

```python
class LVRHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x
```

```python
class LVRHeadGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
```

## 2) 最小 policy / objective 定义

### A. LVR 前向与目标组合
- 文件: `src/train/monkey_patch_forward_lvr.py`
- 符号: `qwen2_5_mixed_modality_forward_lvr`
- 核心性: 统一文本+视觉+LVR token 的前向路径，是 LVR 训练目标的计算入口。
- 关键依赖: `self.model.get_image_features`, `set_lvr_loss_fct`

```python
def qwen2_5_mixed_modality_forward_lvr(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
```

### B. GRPO 训练目标承载
- 文件: `src/trainer/grpo_trainer.py`
- 符号: `QwenGRPOTrainer`
- 核心性: 将生成、奖励、参考模型 KL 与优势估计组织为 RL 训练过程。
- 关键依赖: `GRPOTrainer`, `reward_funcs`, `RepeatSampler`

```python
class RepeatSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
```

## 3) 最小 loss / reward 定义

### A. LVR loss 工厂
- 文件: `src/train/monkey_patch_forward_lvr.py`
- 符号: `set_lvr_loss_fct`
- 核心性: 通过配置切换 MSE/MAE/Cosine，对隐式推理状态进行对齐。
- 关键依赖: `MSELoss`, `L1Loss`, `F.cosine_similarity`

```python
def  set_lvr_loss_fct(loss_lvr_fct: str):
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")
```

### B. GRPO reward
- 文件: `src/train/reward_funcs.py`
- 符号: `accuracy_reward`, `format_reward`
- 核心性: 分别提供答案正确性与输出格式约束奖励，构成可加权奖励集合。
- 关键依赖: `math_verify.parse/verify`, `re.match`

```python
def accuracy_reward(completions, assistant, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass
        rewards.append(reward)
    return rewards
```

```python
def format_reward(completions, **kwargs):
    pattern = r"^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

## 4) 最小 dataset / batch 定义

### A. LVR SFT 数据
- 文件: `src/dataset/lvr_sft_dataset.py`
- 符号: `SupervisedDatasetLVR`
- 核心性: 负责图像+BBox 到 LVR token 索引映射，是 LVR 监督训练的数据基础。
- 关键依赖: `processor.image_processor._preprocess`, `make_bbox_masks_rgb`

```python
class SupervisedDatasetLVR(Dataset):
    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDatasetLVR, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
```

### B. GRPO 数据
- 文件: `src/dataset/grpo_dataset.py`
- 符号: `GRPODataset`
- 核心性: 将样本组织成 chat prompt + assistant 格式，直接对接 GRPO 生成与奖励计算。
- 关键依赖: `llava_to_openai`, `get_image_content`, `get_video_content`

```python
class GRPODataset(Dataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        is_video = False
        contents = []

        if "image" in sources:
            image_files = sources["image"]
            ...
            contents.append(get_image_content(image_file, self.image_min_pixel, self.image_max_pixel,
                                             self.image_resized_w, self.image_resized_h))

        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))
        user_prompt = [{"role": "user", "content": contents}]
        data_dict = dict(prompt=user_prompt, assistant=conversations[1])
        return data_dict
```

## 5) 最小 trainer loop 入口定义

### A. LVR 训练入口
- 文件: `src/train/train_lvr.py`
- 符号: `train`
- 核心性: 聚合参数、patch、模型初始化、token 注册、数据模块、trainer 执行。
- 关键依赖: `replace_qwen2_5_with_mixed_modality_forward_lvr`, `QwenLVRSFTTrainer`

```python
def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head

    replace_qwen2_5_with_mixed_modality_forward_lvr(...)
    model = QwenWithLVR.from_pretrained(model_pth, config=config, torch_dtype=compute_dtype, ...)
    processor = AutoProcessor.from_pretrained(model_args.model_id, ...)
    trainer = QwenLVRSFTTrainer(model=model, processing_class=processor, args=training_args, **data_module)
    trainer.train()
```

### B. LVR 分层优化器
- 文件: `src/trainer/lvr_trainer.py`
- 符号: `QwenLVRSFTTrainer.create_optimizer`
- 核心性: 把视觉塔、merger、lvr_head 与其余参数分组设置学习率，是训练稳定性的关键开关。
- 关键依赖: `get_parameter_names`, `ALL_LAYERNORM_LAYERS`

```python
def create_optimizer(self):
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    if self.args.vision_lr is not None:
        visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
    if self.args.merger_lr is not None:
        merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
    if self.args.lvr_head_lr is not None:
        lvr_head_parameters = [name for name, _ in opt_model.named_parameters() if "lvr_head" in name]

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return self.optimizer
```

## Core Graph
- `src/train/train_lvr.py::train` -> `src/model/qwen_lvr_model.py::QwenWithLVR` (构建带 LVR 的主模型)
- `src/model/qwen_lvr_model.py::QwenWithLVR` -> `src/model/lvr_heads.py::{LVRHead,LVRHeadGLU}` (注入推理头)
- `src/train/train_lvr.py::train` -> `src/dataset/lvr_sft_dataset.py::SupervisedDatasetLVR` (构建 LVR 训练样本)
- `src/train/monkey_patch_forward_lvr.py::qwen2_5_mixed_modality_forward_lvr` -> `src/train/monkey_patch_forward_lvr.py::set_lvr_loss_fct` (计算 LVR 对齐损失)
- `src/train/train_lvr.py::train` -> `src/trainer/lvr_trainer.py::QwenLVRSFTTrainer` (执行监督训练循环)
- `src/train/train_grpo.py::train` -> `src/dataset/grpo_dataset.py::GRPODataset` (构建 GRPO prompt/assistant batch)
- `src/train/train_grpo.py::train` -> `src/train/reward_funcs.py::{accuracy_reward,format_reward}` (装载奖励函数)
- `src/train/train_grpo.py::train` -> `src/trainer/grpo_trainer.py::QwenGRPOTrainer` (执行 RL 生成与优化)
