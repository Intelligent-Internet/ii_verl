# II-VERL

This fork of the [Verl](https://github.com/volcengine/verl) repository introduces several enhancements to improve usability and flexibility when training large language models (LLMs) using reinforcement learning (RL).

We sincerely appreciate the original authors of Verl for their excellent work in developing a robust RL framework for LLMs!

## Improvements

### 1. Huggingface Dataset Integration
- Added support for directly loading datasets from the [Huggingface Hub](https://huggingface.co/datasets).

### 2. YAML-Based Configuration
- Migrated from CLI argument parsing to YAML-based configuration for better readability and maintainability.
- Example of the new configuration format:

#### Old CLI-based Training Command
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
```

#### New YAML-Based Configuration
```yaml
algorithm:
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.001

data:
  train_files: "data/gsm8k/train.parquet"
  val_files: "data/gsm8k/test.parquet"
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: Qwen/Qwen2-7B-Instruct
    use_remove_padding: true
    enable_gradient_checkpointing: true
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 40
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
  rollout:
    log_prob_micro_batch_size_per_gpu: 40
    tensor_model_parallel_size: 2
    name: vllm
    gpu_memory_utilization: 0.6
    n: 5
  ref:
    log_prob_micro_batch_size_per_gpu: 40

trainer:
  critic_warmup: 0
  project_name: "verl_grpo_example_gsm8k"
  experiment_name: "qwen2_7b_function_rm"
  n_gpus_per_node: 8
  nnodes: 1
  save_freq: -1
  test_freq: 5
  total_epochs: 15
```

#### Updated Training Command
```bash
verl train-ppo --config_path path_to_config.yaml --backend fsdp/megatron
```

note: default backend is fsdp

### 3. Remote Reward Server
- Reward calculation logic is now implemented on a remote server.
- During training, LLM outputs and ground-truth values are sent to the server for reward computation.
- The reward server configuration is customizable using the following YAML settings:

```yaml
reward_api:
  enable: true # enable remote reward server
  api_url: http://localhost:8000
  max_workers: 20 # number of concurrent API calls
  timeout: 30 # timeout for each request
  verification_info_column: verification_info # column name of the verification info
  save_dir: debug_data/ # directory to save the reward computation results for easy debugging
  save_freq: 1 # save frequency
```

## Installation
```bash
conda create -n verl python==3.10
conda activate verl

pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

git clone --branch ii_verl https://github.com/Intelligent-Internet/ii_verl.git
cd ii_verl
pip install -e .

# Optional: If you want to use our remote reward server feature
git clone https://github.com/Intelligent-Internet/ii-thought.git
cd ii-thought/rl_verifier
pip install -e .
```

## Usage
### 1. Prepare your dataset (local or Hugging Face)

#### Dataset Format
If you want to use our [reward server](https://github.com/Intelligent-Internet/ii-thought/tree/main/rl_verifier), your dataset must contain two columns:

1. `messages`: This contains the problem or task on which to fine-tune your model. It should be a list of message objects following the standard chat format.
2. `verification_info`: A JSON-parsarable string containing the ground-truth information needed to verify the LLM's output. See [here](https://github.com/Intelligent-Internet/ii-thought/tree/main/rl_verifier#verification-types) for full list of supported verification types.

#### Example Dataset Entry
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Find the largest positive integer \\( x \\) such that \\( x \\) is divisible by all the positive integers \\( \\leq \\sqrt[3]{x} \\)."
    }
  ],
  "verification_info": "{\"answer\": {\"value\": \"420\"}, \"type\": \"math_verifiable\"}"
}
```

If you're not using our reward server, please follow the original verl [instructions](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) for dataset preparation.

### 2. Set up the reward server (for remote reward computation)

#### Basic Setup
Follow the guide in our [repository](https://github.com/Intelligent-Internet/ii-thought) to start the reward server or customize your own. Once running, you can configure it in your YAML file.

#### Example Configuration
```yaml
reward_api:
  enable: true                    # Enable remote reward server
  api_url: http://localhost:8000  # URL of your reward server
  max_workers: 20                 # Number of concurrent API calls
  timeout: 30                     # Timeout in seconds for each request
  verification_info_column: verification_info  # Column name containing verification info
  save_dir: debug_data/           # Directory for saving reward computation results
  save_freq: 1                    # Save frequency (1 = save after each computation)
```

### 3. Define your configuration in a YAML file
See the example configurations in `examples/yaml/` for detailed setup options.

### 4. Start training
```bash
verl train-ppo config
```

## License
This project follows the same license as the original Verl repository.