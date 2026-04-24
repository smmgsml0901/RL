# ME5406 Project
## Overview

This project trains a FRANKA robotic arm (6-DoF, wrist joints locked) to navigate its end-effector through a narrow aperture in a rigid plate and reach a target on the far side。

## Structure

project/
├── PPO/
├── SAC/
├── TD3/
├── common/
│   ├── env.py
│   ├── normalization.py
│   └── __init__.py
├── requirements.txt
└── README.md



## Operation

### Prerequisites
- Ubuntu 20.04 / 22.04
- NVIDIA GPU with CUDA 11.8+ (RTX 3090 or better recommended for fast training)
Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### PPO training

```bash
python PPO/train.py --stage 1
```

### PPO assessment

```bash
python PPO/eval_actor_only.py --checkpoint checkpoints/stage1/<run_id>/best.pt --episodes 10
```

### SAC training

```bash
python SAC/wbh_sac.py --stage 1
```

### SAC assessment

```bash
python SAC/eval_sac.py --checkpoint checkpoints/stage1/<run_id>/best.pt --episodes 10
```
### TD3 training

```bash
python TD3/train_TD3.py --stage 1
```
### TD3 assessment

```bash
python TD3/eval_actor_only.py --checkpoint checkpoints/stage1/<run_id>/best.pt --episodes 10
```
