export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/work/marzieh/huggingface


# Full training:
accelerate launch --main_process_port 29600 --config_file configs/single_gpu.yaml examples/train_kd.py  --config_file configs/kronecker_kd.json
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


    
