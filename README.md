# Finetuning in Gaudi using LoRA and Deepspeed

## Introduction

In this repository, we explore techniques for fine-tuning Large Language Models (LLMs) using:

1. **LoRA** (Low-Rank Adaptation)
2. **DeepSpeed**
3. **Custom Trainer** optimized for Gaudi hardware.

## Getting Started

### General Requirements

Inside the Gaudi docker environment, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

To initiate the fine-tuning process, execute the following command:

```bash
PT_HPU_LAZY_MODE=0 python finetune.py --config-name=finetune_lora.yaml
```

**Explanation:**
- `PT_HPU_LAZY_MODE=0`: Enables eager mode as lazy mode is currently not supported on Gaudi.
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
