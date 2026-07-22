# Unitree G1 Humanoid Standup

📄 **[View the Project Report](./report/report.md)**

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/mohitjavale/takehome_assignment.git
cd takehome_assignment
```

**2. Add Isaac Gym:**
Download Isaac Gym (Preview 4) from [NVIDIA's website](https://developer.nvidia.com/isaac-gym) and extract it directly into the root of this project.  
No need to install it, the uv path has been set to 
Your directory structure must match this layout:
```text
takehome_assignment/
├── isaacgym/
│   ├── python/
│   └── ...
├── report/
├── src/
└── ...
```

**3. Install Dependencies:**  
Install `uv` if you don't already have it.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use `uv` to sync and install all required packages:
```bash
uv sync
```

---



### Training
Start training a new PPO policy:
```bash
cd src/humanoid_standup/scripts/
uv run train.py training.exp_name="<your_experiment_name>"
```

### Evaluation
Evaluate a trained policy. The scripts automatically load the latest checkpoint for the specified experiment name.

**Standard Evaluation:**
Runs the normal environment baseline.
```bash
cd src/humanoid_standup/scripts/
uv run eval.py training.exp_name="<your_experiment_name>"
```

**Robustness Evaluation (`eval_lie.py`):**
Tests out-of-distribution recovery by initially pushing zero actions (letting the robot fall to the ground) and calculates the final stand-up success rate.
```bash
cd src/humanoid_standup/scripts/
uv run eval_lie.py training.exp_name="<your_experiment_name>"
```

---

## Pre-trained Checkpoint

A successful pre-trained policy checkpoint (`t10`) is included in the repository for immediate testing. 

To evaluate it, navigate to the `scripts` directory and run either evaluation script:
```bash
cd src/humanoid_standup/scripts/
uv run eval.py training.exp_name="t10"
# OR
uv run eval_lie.py training.exp_name="t10"
```