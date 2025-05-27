export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/work/marzieh/huggingface
export WANDB_PROJECT="kronecker"
export WANDB_ENTITY="nlp_markham"
export WANDB_API_KEY="${WANDB_API_KEY}"
export PYTHONPATH=/work/marzieh/kronecker/kronecker-kd:$PYTHONPATH
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"

# Full training:
accelerate launch --main_process_port 29605 --config_file configs/single_gpu.yaml examples/train_kd.py --config configs/kronecker_kd.json
    # --model_name_or_path Qwen/Qwen3-0.6B \
    # --teacher_model_name_or_path Qwen/Qwen3-0.6B \
    # --dataset_name JunxiongWang/sftdatasetv3 \    --learning_rate 2e-5 \
    # --per_device_train_batch_size 4 \
    # --gradient_accumulation_steps 8 \
    # --output_dir gkd-model \
    # --logging_steps 10 \
    # --num_train_epochs 1 \
    # --push_to_hub \
    # --gradient_checkpointing \
    # --max_length 2048 \


    
