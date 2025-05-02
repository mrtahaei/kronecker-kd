export CUDA_VISIBLE_DEVICES = 0,1
export HF_HOME = /work/marzieh/huggingface

# Full training:
python examples/train_kd.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gkd-model \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing