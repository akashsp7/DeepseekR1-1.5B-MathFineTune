#!/usr/bin/env python3
"""
DeepSeek R1 Fine-tuning for Mathematical Reasoning
Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B on mathematical reasoning tasks using LoRA.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset


def create_dataset(original_dataset):
    """Create balanced dataset from original with equal representation of difficulties"""
    # Convert to pandas DataFrame for faster processing
    df = original_dataset['train'].to_pandas()

    # Initial filtering
    mask = (
        df['difficulty'].isin(['easy', 'medium', 'hard', 'very hard']) &
        (df['task_category'] == 'Math') &
        df['response'].str.contains('</think>', na=False) &
        df.notna().all(axis=1)
    )
    filtered_df = df[mask]

    # Separate very hard samples
    very_hard_df = filtered_df[filtered_df['difficulty'] == 'very hard']

    # Get main difficulties
    main_df = filtered_df[filtered_df['difficulty'].isin(['easy', 'medium', 'hard'])]

    # Get counts and minimum
    difficulty_counts = main_df['difficulty'].value_counts()
    min_count = min(difficulty_counts[['easy', 'medium', 'hard']])
    
    # Sample equal numbers from each difficulty
    balanced_dfs = []
    for diff in ['easy', 'medium', 'hard']:
        diff_df = main_df[main_df['difficulty'] == diff]
        balanced_dfs.append(diff_df.sample(n=min_count, random_state=42))
    
    # Combine all dataframes
    final_df = pd.concat(balanced_dfs + [very_hard_df], ignore_index=True)

    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = final_df[['instruction', 'response', 'intent', 'knowledge', 'difficulty']]
    
    print("Final difficulty distribution:")
    print(final_df['difficulty'].value_counts())

    # Convert back to HuggingFace dataset
    final_ds = Dataset.from_pandas(final_df)

    del df, filtered_df, very_hard_df, main_df, balanced_dfs, difficulty_counts, min_count
    return final_ds


def split_response(example):
    """Split response into thinking and final answer parts"""
    parts = example['response'].split('</think>')

    # Get the thinking part (remove <think> tag and strip whitespace)
    thinking = parts[0].replace('<think>', '').strip()

    # Get the response part (everything after </think>, or empty string if nothing after)
    response = parts[1].strip() if len(parts) > 1 else ""

    return {
        'thinking': thinking,
        'response': response
    }


def formatting_prompts_func(examples):
    """Format examples into training prompts"""
    questions = examples["instruction"]
    intent = examples["intent"]
    knowledge = examples["knowledge"]
    thinking = examples["thinking"]
    response = examples["response"]

    train_prompt_style = """Below is an instruction that describes a mathematical task, paired with additional context information to guide the solution.
Write a response that thoroughly solves the given problem.
Before solving, develop a clear step-by-step chain of reasoning to ensure accuracy and logical coherence.

### Instruction:
You are a mathematics expert with advanced knowledge in mathematical reasoning, problem-solving, and proof techniques. You think outloud and consider various aspects before giving any concrete answers.

### Question:
{}

### Intent:
{}

### Knowledge Required:
{}

### Response:
<think>
{}
</think>
{}"""

    texts = []
    for q, i, k, t, r in zip(questions, intent, knowledge, thinking, response):
        text = train_prompt_style.format(q, i, k, t, r) + tokenizer.eos_token
        texts.append(text)

    return {"text": texts}


def split_dataset(dataset, test_size=0.1, val_size=0.1, random_state=42):
    """Split dataset into train, test, and validation sets while stratifying by difficulty"""
    # Get initial difficulty distribution
    difficulty_counts = Counter(dataset['difficulty'])
    print("Original distribution:")
    for difficulty, count in difficulty_counts.items():
        print(f"{difficulty}: {count}")

    # Create indices for splitting
    indices = list(range(len(dataset)))
    difficulties = dataset['difficulty']

    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=difficulties,
        random_state=random_state
    )

    # Second split: separate validation set from training set
    adjusted_val_size = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_size,
        stratify=[difficulties[i] for i in train_val_indices],
        random_state=random_state
    )

    # Create the datasets with all columns first for distribution checking
    train_ds_full = dataset.select(train_indices)
    test_ds_full = dataset.select(test_indices)
    val_ds_full = dataset.select(val_indices)

    # Print distributions
    print("\nTrain set distribution:")
    train_counts = Counter(train_ds_full['difficulty'])
    for difficulty, count in train_counts.items():
        print(f"{difficulty}: {count}")

    print("\nTest set distribution:")
    test_counts = Counter(test_ds_full['difficulty'])
    for difficulty, count in test_counts.items():
        print(f"{difficulty}: {count}")

    print("\nValidation set distribution:")
    val_counts = Counter(val_ds_full['difficulty'])
    for difficulty, count in val_counts.items():
        print(f"{difficulty}: {count}")

    # Create the final datasets with only the 'text' column
    train_ds = train_ds_full.remove_columns([col for col in dataset.column_names if col != 'text'])
    test_ds = test_ds_full.remove_columns([col for col in dataset.column_names if col != 'text'])
    val_ds = val_ds_full.remove_columns([col for col in dataset.column_names if col != 'text'])

    # Print final sizes
    print(f"\nFinal sizes:")
    print(f"Train set size: {len(train_ds)}")
    print(f"Test set size: {len(test_ds)}")
    print(f"Validation set size: {len(val_ds)}")

    return train_ds, test_ds, val_ds


def test_model_inference(model, tokenizer, test_question):
    """Test model inference with a sample question"""
    prompt_style = """Below is an instruction that describes a mathematical task, paired with additional context information to guide the solution.
Write a response that thoroughly solves the given problem.
Before solving, develop a clear step-by-step chain of reasoning to ensure accuracy and logical coherence.

### Instruction:
You are a mathematics expert with advanced knowledge in mathematical reasoning, problem-solving, and proof techniques. You think outloud and consider various aspects before giving a concrete answers.

### Question:
{}

### Response:
<think>{}"""

    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(test_question, "")], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4096,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)
    print("Model Response:")
    print(response[0].split("### Response:")[1])


def main():
    # Configuration
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    batch_size = 2
    gradient_steps = 8
    
    # Set model output directory
    output_dir = "./deepseek_finetuned_model"
    
    print("Loading dataset...")
    ds = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B")
    
    print("Creating balanced dataset...")
    filtered_ds = create_dataset(ds)
    
    # Clean up original dataset
    ds.cleanup_cache_files()
    del ds
    
    print("Loading model and tokenizer...")
    global model, tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Test model before training
    test_question = '''A snail travels at 1 cm per second for 1 minute, then teleports 10 meters backward every 30 seconds for 3 minutes while a turtle moving at 0.5 cm/s chases it.
How far is the snail from its starting point after 3 minutes?'''
    
    print("\n=== Testing model before training ===")
    test_model_inference(model, tokenizer, test_question)
    
    print("\nProcessing dataset for training...")
    # Split response into thinking and answer parts
    filtered_ds = filtered_ds.map(split_response)
    
    # Format for training
    finetuning_data = filtered_ds.map(formatting_prompts_func, batched=True)
    
    # Split dataset
    train_ds, test_ds, val_ds = split_dataset(finetuning_data, test_size=0.05, val_size=0.05, random_state=42)
    
    print("Setting up LoRA...")
    # Apply LoRA fine-tuning to the model
    model_lora = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )
    
    print(f"Original model parameters: {model.num_parameters():,}")
    print(f"LoRA model parameters: {model_lora.num_parameters():,}")
    
    # Clean up inference optimization before training
    if hasattr(model_lora, "_unwrapped_old_generate"):
        try:
            if model_lora._unwrapped_old_generate is not None:
                model_lora._unwrapped_old_generate = None
            if hasattr(model_lora, "generate") and hasattr(model, "generate"):
                model_lora.generate = model.generate
                print("Successfully restored original generate method")
        except AttributeError as e:
            print(f"Warning: Could not fully clean up _unwrapped_old_generate: {e}")
    
    steps_per_epoch = len(train_ds) / (batch_size * gradient_steps)
    print(f"Steps per epoch: {int(steps_per_epoch)}")
    
    print("Starting training...")
    # Define training arguments
    trainer = SFTTrainer(
        model=model_lora,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_steps,
            num_train_epochs=2,
            warmup_ratio=0.1,
            learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
            report_to=None,  # Disable wandb for general use
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    print("\n=== Testing model after training ===")
    test_model_inference(model_lora, tokenizer, test_question)
    
    print(f"\nSaving model to {output_dir}...")
    model_lora.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
