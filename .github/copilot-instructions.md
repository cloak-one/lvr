# Project Guidelines

## Environment
- Required runtime environment: conda environment named train.
- Before running any Python command, training script, evaluation script, or tests, always activate train first.
- Standard command prefix:
  - source /root/miniconda3/etc/profile.d/conda.sh && conda activate train
- If a command fails and environment is unclear, first print environment diagnostics:
  - which python
  - python -V
  - python -c "import sys; print(sys.executable)"

## Architecture
- Main code is under src with clear module boundaries:
  - src/model: model definitions and LVR heads
  - src/dataset: dataset preparation for SFT/LVR/DPO/GRPO
  - src/train: stage entrypoints and monkey patches
  - src/trainer: trainer implementations for each training paradigm
- LVR core logic is implemented by combining model extensions and forward monkey patching.
- Prefer understanding pipeline from these docs instead of re-describing architecture in chat:
  - docs/document-b-macro-structure-workflow.md
  - docs/document-a-core-micro-view.md

## Build And Test
- There is no unified build system; the repo is script-driven.
- Common execution patterns:
  - Stage 1 training: bash scripts/finetune_lvr_stage1_7b.sh
  - Stage 2 GRPO training: bash scripts/finetune_lvr_stage2_7b.sh
  - Evaluation: python evaluation/evaluation.py
- Always run commands from repo root with PYTHONPATH configured as needed by scripts.
- For quick smoke checks after code edits, prefer lightweight import checks over full training runs.

## Conventions
- Keep changes minimal and localized; do not refactor unrelated modules.
- Avoid relying on private/internal transformers symbols. Prefer stable public APIs to reduce version fragility.
- Reward functions are discovered dynamically by name pattern in src/train/reward_funcs.py. New reward functions should follow the existing naming convention ending with _reward.
- Keep monkey patch behavior consistent with the target training stage; patch files in src/train are stage-specific.

## Safety And Ops
- Do not commit or expose credentials. evaluation/evaluation.py currently contains hard-coded cloud credentials; treat this as a risk area and prefer environment variables for new code.
- For distributed runs, keep deepspeed config paths and stage scripts aligned; do not change ZeRO settings unless task requires it.

## Key References
- Model entry: src/model/qwen_lvr_model.py
- GRPO entry: src/train/train_grpo.py
- LVR forward patch: src/train/monkey_patch_forward_lvr.py
- GRPO forward patch: src/train/monkey_patch_forward_lvr_rl.py
- Reward utilities: src/utils.py and src/train/reward_funcs.py
