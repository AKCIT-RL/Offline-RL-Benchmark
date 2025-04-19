# Offline RL Benchmark

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended)

### Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Offline-RL-Benchmark.git
cd Offline-RL-Benchmark
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install mujoco-playground:

```bash
git clone git@github.com:AKCIT-RL/mujoco_playground.git
cd mujoco_playground
pip install -e ".[all]"
```




## Usage

### Random

```bash
python collect_data.py --env_name <ENV_NAME> --description <DESCRIPTION> --difficulty "random" --num_envs 100 --model_checkpoint "checkpoints/<CHECKPOINT_NAME>"
```

### Medium 

```bash
python collect_data.py --env_name <ENV_NAME> --description <DESCRIPTION> --difficulty "medium" --num_envs 10 --model_checkpoint "checkpoints/<CHECKPOINT_NAME>"
```
### Expert

```bash
python collect_data.py --env_name <ENV_NAME> --description <DESCRIPTION> --difficulty "expert" --num_envs 100 --model_checkpoint "checkpoints/<CHECKPOINT_NAME>"
```

