# Environment Setup

## Required Environment Variables

Before running the training script `scripts/train_kd.sh`, you need to set the following environment variables:

### 1. Weights & Biases API Key
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

### 2. Hugging Face Hub Token
```bash
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token_here"
```

## Setup Options

### Option 1: Set in your shell session
```bash
export WANDB_API_KEY="your_actual_key"
export HUGGINGFACE_HUB_TOKEN="your_actual_token"
./scripts/train_kd.sh
```

### Option 2: Create a local environment file
Create a file called `.env` (which should be gitignored):
```bash
# .env file
WANDB_API_KEY=your_actual_key
HUGGINGFACE_HUB_TOKEN=your_actual_token
```

Then source it before running the script:
```bash
source .env
./scripts/train_kd.sh
```

### Option 3: Set inline with the script
```bash
WANDB_API_KEY="your_key" HUGGINGFACE_HUB_TOKEN="your_token" ./scripts/train_kd.sh
```

## Security Note

Never commit actual API keys or tokens to version control. The training script has been updated to use environment variables to prevent accidental exposure of sensitive credentials. 