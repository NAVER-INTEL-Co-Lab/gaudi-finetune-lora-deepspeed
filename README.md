# Finetuning in Gaudi-v2 using LoRA and Deepspeed

## Introduction

In this repository, we explore techniques for fine-tuning Large Language Models (LLMs) using:

1. **LoRA** (Low-Rank Adaptation)
2. **DeepSpeed**
3. **Custom Trainer** optimized

for Gaudi-v2 using the habana framework.

## Getting Started

### General Requirements

Inside the Gaudi-v2 docker environment, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

To initiate the fine-tuning process, execute the following command:

```bash
PT_HPU_LAZY_MODE=0 python finetune.py --config-name=finetune_lora.yaml
PT_HPU_LAZY_MODE=0 python gaudi_spawn.py --world_size 7 --use_deepspeed finetune.py --config-name=finetune_lora.yaml
```

**Explanation:**
- `PT_HPU_LAZY_MODE=0`: Enables eager mode as lazy mode is currently not supported on Gaudi-v2.
- `--world_size 7`: Amount of HPU used
- `--use_deepspeed`: Whether using deepspeed or not
- `--config-name`: Specifies the configuration file. Detailed configurations can be found in `config/finetune_lora.yaml`.

### LoRA Configuration

You can customize the LoRA parameters (e.g., rank, scaling factor, and dropout rate) in the configuration file `config/finetune_lora.yaml`:

```yaml
LoRA:
  r: 8        # Rank of the LoRA matrices
  alpha: 32   # Scaling factor
  dropout: 0.05 # Dropout probability
```

### DeepSpeed Configuration

DeepSpeed parameters can be configured in the file `config/ds_config.json`. Example:

```json
{
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {
      "device": "none",
      "pin_memory": true
    }
  }
}
```
This configuration enables stage-1 ZeRO optimization without offloading the optimizer. Feel free to use stage-2 or stage-3 optimization.

### Custom Trainer

An example of the custom trainer implementation can be found in `dataloader.py`. This code shows how to have a custom trainer wrapped by the GaudiTrainer class.


## Detailed Explanation

### finetune.py

The [finetune.py](http://finetune.py) code is the main code of the repository. It contains the training arguments using `GaudiTrainingArguments`:

```python
training_args = GaudiTrainingArguments(
    use_habana=True,
    use_lazy_mode=False,
    gaudi_config_name=cfg.gaudi_config_name,

    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    ...
    deepspeed='config/ds_config.json', # DeepSpeed configuration using built-in args in Transformers
    weight_decay=cfg.weight_decay,
    seed=cfg.seed,
)
```

Since `GaudiTrainingArguments` is derived from Hugging Face’s `TrainingArguments`, we can easily integrate DeepSpeed here.

The code for LoRA is also written here, leveraging the PEFT library:

```python
# LoRA configuration
if cfg.LoRA.r != 0:
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
```

The LoRA setup is processed before the training begins.

Finally, we start the training process using the custom trainer:

```python
# Training using CustomTrainer from dataloader.py
trainer = CustomTrainer(
    model=model,
    train_dataset=ft_dataset,
    eval_dataset=ft_dataset,
    args=training_args,
    data_collator=custom_data_collator,
)
model.config.use_cache = False  # Silence the warnings.
trainer.train()
```

### dataloader.py

In `dataloader.py`, we implemented an example of a custom trainer using the `GaudiTrainer` instance. This is useful for implementing custom training processes such as unlearning:

```python
import torch
from optimum.habana import GaudiTrainer

class CustomTrainer(GaudiTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
```

### Config Folders

The `config` folder contains parameter specifications for the training process, LoRA, and DeepSpeed.

#### config/finetune_lora.yaml

```yaml
model_id: NousResearch/Llama-2-7b-chat-hf
model_family: llama2-7b
gaudi_config_name: Habana/llama

LoRA:
  r: 8
  alpha: 32
  dropout: 0.05

data_path: ../path/to/finetune/data
split: full
batch_size: 4
gradient_accumulation_steps: 1
num_epochs: 10
save_dir: /save_dir
lr: 1e-4
weight_decay: 0
seed: 42
```

#### config/ds_config.json

```json
{
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
...
        }
    }
}
```
