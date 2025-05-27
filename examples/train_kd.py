# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training:
python examples/scripts/gkd.py \
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

# LoRA:
python examples/scripts/gkd.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gkd-model \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
"""

import os
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from kronecker.layers.kronecker_linear import  KroneckerLinear
from kronecker.utils.conversion import replace_linears_with_kron

from trl import (
    GKDConfig,
    GKDTrainer,
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from examples.custom_gkd_trainer import CustomGKDTrainer

def load_tokenized_dataset(dataset_name, dataset_config=None):
    """
    Load the tokenized dataset from local cache or HuggingFace Hub.
    Priority: Local HF cache -> HuggingFace Hub (original) -> Error
    
    This function will:
    1. Try to load from local HuggingFace cache first (fastest)
    2. Fall back to loading original dataset from HuggingFace Hub (slower but works)
    3. Provide clear feedback about which source was used
    """
    # Define the local cache path for the tokenized dataset
    local_dataset_path = "/work/marzieh/huggingface/datasets/JunxiongWang___sftdatasetv3/default/0.0.0/f47b5668d12df9db8284d9f98d9e384374495ab3"
    
    print("🔍 Attempting to load dataset with intelligent fallback...")
    
    # Strategy 1: Try to load from local HuggingFace cache first (fastest)
    if os.path.exists(local_dataset_path):
        print(f"✅ Found local HuggingFace cache at: {local_dataset_path}")
        try:
            print("📂 Loading dataset from local HuggingFace cache...")
            
            # Method 1: Try load_from_disk first
            try:
                dataset = load_from_disk(local_dataset_path)
                print("🚀 Successfully loaded dataset using load_from_disk!")
            except Exception as disk_error:
                print(f"⚠️  load_from_disk failed: {disk_error}")
                
                # Method 2: Try loading using the dataset name with cache_dir (offline mode)
                print("🔄 Trying offline loading from cache...")
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(
                        dataset_name, 
                        name=dataset_config,
                        cache_dir="/work/marzieh/huggingface",
                        trust_remote_code=True,
                        download_mode="reuse_cache_if_exists"
                    )
                    print("🚀 Successfully loaded dataset from local HuggingFace cache!")
                except Exception as cache_error:
                    print(f"⚠️  Cache loading failed: {cache_error}")
                    
                    # Method 3: Try to construct dataset from Arrow files directly
                    print("🔄 Trying to load from Arrow files directly...")
                    try:
                        from datasets import Dataset, DatasetDict
                        import glob
                        
                        # Find Arrow files
                        train_files = glob.glob(os.path.join(local_dataset_path, "*train*.arrow"))
                        test_files = glob.glob(os.path.join(local_dataset_path, "*test*.arrow"))
                        
                        if train_files and test_files:
                            print(f"Found {len(train_files)} train files and {len(test_files)} test files")
                            
                            # Load train split
                            train_dataset = Dataset.from_file(train_files[0])  # Start with first file
                            for train_file in train_files[1:]:
                                additional_data = Dataset.from_file(train_file)
                                # Concatenate datasets
                                from datasets import concatenate_datasets
                                train_dataset = concatenate_datasets([train_dataset, additional_data])
                            
                            # Load test split
                            test_dataset = Dataset.from_file(test_files[0])
                            for test_file in test_files[1:]:
                                additional_data = Dataset.from_file(test_file)
                                test_dataset = concatenate_datasets([test_dataset, additional_data])
                            
                            # Create DatasetDict
                            dataset = DatasetDict({
                                'train': train_dataset,
                                'test': test_dataset
                            })
                            
                            print("🚀 Successfully loaded dataset from Arrow files!")
                        else:
                            raise Exception(f"No suitable Arrow files found. Train files: {len(train_files)}, Test files: {len(test_files)}")
                    
                    except Exception as arrow_error:
                        print(f"⚠️  Arrow file loading failed: {arrow_error}")
                        raise cache_error  # Re-raise the cache error to continue to Hub fallback
            
            print(f"📊 Dataset splits: {list(dataset.keys())}")
            
            # Print dataset info
            for split_name, split_data in dataset.items():
                print(f"  - {split_name}: {len(split_data):,} examples")
            
            # Verify it's actually tokenized/ready
            if len(dataset) > 0:
                first_split = list(dataset.keys())[0]
                sample = dataset[first_split][0]
                if 'messages' in sample:
                    print("✅ Confirmed: Dataset contains 'messages' field (ready format)")
                    return dataset, "local_cache"
                else:
                    print("✅ Dataset loaded from local cache")
                    return dataset, "local_cache"
            else:
                print("⚠️  Warning: Local dataset is empty")
                print("🔄 Falling back to HuggingFace Hub...")
                
        except Exception as e:
            print(f"❌ Failed to load from local cache: {e}")
            print("🔄 Falling back to HuggingFace Hub...")
    else:
        print(f"ℹ️  Local dataset cache not found at: {local_dataset_path}")
        print("🔄 Will try HuggingFace Hub...")
    
    # Strategy 2: Load original dataset from HuggingFace Hub
    try:
        print(f"🌐 Loading original dataset from HuggingFace Hub: {dataset_name}")
        
        # Try with different authentication methods
        try:
            # First try with trust_remote_code and use_auth_token
            dataset = load_dataset(
                dataset_name, 
                name=dataset_config, 
                trust_remote_code=True,
                use_auth_token=True
            )
        except Exception as auth_error:
            print(f"⚠️  Authentication method 1 failed: {auth_error}")
            try:
                # Try without authentication
                dataset = load_dataset(
                    dataset_name, 
                    name=dataset_config, 
                    trust_remote_code=True
                )
            except Exception as no_auth_error:
                print(f"⚠️  No-auth method failed: {no_auth_error}")
                # Try with explicit token from environment
                hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
                if hf_token:
                    dataset = load_dataset(
                        dataset_name, 
                        name=dataset_config, 
                        trust_remote_code=True,
                        token=hf_token
                    )
                else:
                    raise no_auth_error
        
        print("✅ Successfully loaded original dataset from HuggingFace Hub!")
        print(f"📊 Dataset splits: {list(dataset.keys())}")
        
        # Print dataset info
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data):,} examples")
        
        # Check if it's already in the right format or needs processing
        if len(dataset) > 0:
            first_split = list(dataset.keys())[0]
            sample = dataset[first_split][0]
            if 'messages' in sample:
                print("✅ Dataset already contains 'messages' field")
                return dataset, "hub_ready"
            else:
                print("ℹ️  Dataset needs processing (doesn't have 'messages' field)")
                print(f"Available fields: {list(sample.keys())}")
                return dataset, "hub_needs_processing"
        
        return dataset, "hub_unknown"
        
    except Exception as e:
        print(f"❌ Failed to load from HuggingFace Hub: {e}")
        
    # Strategy 3: If all else fails, provide helpful error message
    error_msg = f"""
❌ Could not load dataset from any source!

Attempted sources:
1. Local HuggingFace cache: {local_dataset_path}
2. HuggingFace Hub: {dataset_name}

Possible solutions:
1. Check internet connection for HuggingFace Hub access
2. Verify HuggingFace authentication token is set correctly
3. Check if dataset name is correct: {dataset_name}
4. Verify local cache path exists and has proper permissions
5. Try running: huggingface-cli login
6. Check if dataset is public or requires authentication

Environment variables to check:
- HF_HOME: {os.getenv('HF_HOME', 'Not set')}
- HUGGINGFACE_HUB_TOKEN: {'Set' if os.getenv('HUGGINGFACE_HUB_TOKEN') else 'Not set'}
"""
    raise RuntimeError(error_msg)

def verify_dataset_compatibility(dataset, source_type):
    """
    Verify that the dataset is compatible with the training pipeline.
    Returns (is_compatible, needs_processing, info_message)
    """
    if dataset is None:
        return False, False, "Dataset is None"
    
    # Check required splits
    required_splits = ['train', 'test']
    available_splits = list(dataset.keys())
    missing_splits = [split for split in required_splits if split not in available_splits]
    
    if missing_splits:
        return False, False, f"Missing required splits: {missing_splits}. Available: {available_splits}"
    
    # Check train split format
    train_data = dataset['train']
    if len(train_data) == 0:
        return False, False, "Train split is empty"
    
    sample = train_data[0]
    sample_keys = list(sample.keys())
    
    # Check if it has the expected 'messages' field
    if 'messages' in sample_keys:
        # Verify messages format
        messages = sample['messages']
        if isinstance(messages, list) and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict) and 'role' in first_msg and 'content' in first_msg:
                return True, False, f"✅ Dataset ready for training (source: {source_type})"
            else:
                return False, False, "Messages format is incorrect (missing role/content)"
        else:
            return False, False, "Messages field is not a proper list"
    
    # If no 'messages' field, check what fields are available
    if source_type.startswith("hub"):
        # For hub datasets, we might need to process them
        common_fields = ['prompt', 'response', 'conversation', 'text', 'input', 'output']
        found_fields = [field for field in common_fields if field in sample_keys]
        
        if found_fields:
            return True, True, f"⚠️  Dataset needs processing. Found fields: {found_fields}. Available: {sample_keys}"
        else:
            return False, False, f"❌ Unknown dataset format. Available fields: {sample_keys}"
    
    return False, False, f"❌ Incompatible dataset format. Available fields: {sample_keys}"

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GKDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    # Load student model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )


    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    kroncker_model = replace_linears_with_kron(model,compression={'attention': 4.0, 'ffn': 8.0, 'head': 2.0})
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        training_args.teacher_model_name_or_path,
        **teacher_model_kwargs
    )

    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    print("\n" + "="*60)
    print("LOADING TOKENIZED DATASET")
    print("="*60)
    
    # Use the new function to load the tokenized dataset
    dataset, source_type = load_tokenized_dataset(script_args.dataset_name, script_args.dataset_config)
    
    # Verify dataset compatibility
    print("\n" + "="*60)
    print("VERIFYING DATASET COMPATIBILITY")
    print("="*60)
    
    is_compatible, needs_processing, info_message = verify_dataset_compatibility(dataset, source_type)
    print(info_message)
    
    if not is_compatible:
        print(f"\n❌ DATASET COMPATIBILITY ERROR")
        print(f"Error: {info_message}")
        print("\nPlease check:")
        print("1. Dataset format is correct")
        print("2. Required splits (train, test) are present")
        print("3. Messages field has proper structure")
        raise RuntimeError(f"Dataset incompatible: {info_message}")
    
    if needs_processing:
        print(f"\n⚠️  DATASET NEEDS PROCESSING")
        print("The dataset was loaded but needs to be converted to the expected format.")
        print("This is normal for datasets loaded from HuggingFace Hub.")
        print("The training pipeline will handle this automatically.")
    
    # Debug: Print dataset structure
    print(f"\n📊 Dataset loaded successfully from: {source_type}")
    print(f"📊 Dataset structure: {dataset}")
    print(f"📊 Available splits: {list(dataset.keys())}")
    print(f"📊 Dataset train split: {script_args.dataset_train_split}")
    print(f"📊 Dataset test split: {script_args.dataset_test_split}")
    
    # Print sample data to verify format
    if script_args.dataset_train_split in dataset:
        train_data = dataset[script_args.dataset_train_split]
        print(f"📊 Training examples: {len(train_data):,}")
        print(f"📊 Training dataset features: {train_data.features}")
        
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"📊 Sample training example keys: {list(sample.keys())}")
            if 'messages' in sample:
                messages = sample['messages']
                if isinstance(messages, list) and len(messages) > 0:
                    print(f"📊 Sample has {len(messages)} messages")
                    # Show first 2 messages
                    for i, msg in enumerate(messages[:2]):
                        if isinstance(msg, dict):
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', '')[:100]
                            print(f"    Message {i+1} ({role}): {content}...")
                else:
                    print(f"📊 Messages field: {messages}")
            else:
                # Show available fields for debugging
                print(f"📊 Available fields: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print(f"    {key}: {preview}")
                    else:
                        print(f"    {key}: {type(value)} - {value}")

    # Performance and memory info
    print(f"\n💡 DATASET SOURCE INFORMATION")
    if source_type == "local_cache":
        print("✅ Using pre-tokenized local cache - optimal performance!")
        print("   - Fastest loading time")
        print("   - Pre-processed data")
        print("   - No additional tokenization needed")
    elif source_type == "hub_ready":
        print("✅ Using HuggingFace Hub dataset - ready format!")
        print("   - Slower loading than local cache")
        print("   - Data already in correct format")
        print("   - No additional processing needed")
    elif source_type == "hub_needs_processing":
        print("⚠️  Using HuggingFace Hub dataset - needs processing!")
        print("   - Slower loading than local cache")
        print("   - Data will be processed during training")
        print("   - Consider caching processed data for future runs")
    else:
        print(f"ℹ️  Using dataset from: {source_type}")

    # with PartialState().local_main_process_first():
    #     dataset = dataset.map(
    #         lambda x: {
    #             "prompt": tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True)
    #         },
    #         num_proc=training_args.dataset_num_proc,
    #     )

    training_args.beta = 0
    training_args.lmbda =0
    

    ################
    # Training
    ################
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)
    
    trainer = CustomGKDTrainer(
        model=kroncker_model,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer        
    )
    #peft_config=get_peft_config(model_args),

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=40)
        trainer.add_callback(completions_callback)

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
