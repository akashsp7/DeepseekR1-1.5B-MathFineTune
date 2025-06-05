#!/usr/bin/env python3
"""
DeepSeek Model Evaluation Script
Compares the performance of fine-tuned vs base DeepSeek model on mathematical reasoning tasks.
"""

import os
import json
import pickle
import torch
from collections import Counter
from sklearn.model_selection import train_test_split

from unsloth import FastLanguageModel
from datasets import load_dataset


def create_dataset(original_dataset):
    """Create filtered dataset for evaluation"""
    # Filter dataset for high-quality math problems
    filtered_ds = original_dataset['train'].filter(
        lambda x: (
            x['input_quality'] == 'excellent' and
            x['task_category'] == 'Math' and
            '</think>' in x['response'] and
            all(x[field] is not None for field in x.keys())
        )
    )
    
    # Keep only necessary columns
    columns_to_keep = ['instruction', 'response', 'intent', 'knowledge', 'difficulty']
    filtered_ds = filtered_ds.select_columns(columns_to_keep)
    
    return filtered_ds


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
    """Format examples for training with full context"""
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

### Response:

## Intent:
{}

## Knowledge Required:
{}

<think>
{}
</think>
{}"""

    texts = []
    for q, i, k, t, r in zip(questions, intent, knowledge, thinking, response):
        text = train_prompt_style.format(q, i, k, t, r)
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
    test_ds = test_ds_full
    val_ds = val_ds_full.remove_columns([col for col in dataset.column_names if col != 'text'])

    # Print final sizes
    print(f"\nFinal sizes:")
    print(f"Train set size: {len(train_ds)}")
    print(f"Test set size: {len(test_ds)}")
    print(f"Validation set size: {len(val_ds)}")

    return train_ds, test_ds, val_ds


def generate_responses(model, tokenizer, instructions, prompt_template):
    """Generate responses for a list of instructions using the given model"""
    responses = []
    
    for instruction in instructions:
        prompt = prompt_template.format(instruction, "")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=4096,
                use_cache=True,
                do_sample=False,  # Use greedy decoding for consistency
                temperature=0.1,
            )
        
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.append(decoded_output[0])
        
        # Clear GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return responses


def compare_responses(finetuned_outputs, normal_outputs, ground_truth_responses):
    """Compare and display model responses"""
    print("=" * 100)
    print("MODEL COMPARISON RESULTS")
    print("=" * 100)
    
    for i, (tuned, normal, truth) in enumerate(zip(finetuned_outputs, normal_outputs, ground_truth_responses)):
        print(f"\n--- Example {i+1} ---")
        print('--' * 10 + 'Ground Truth Response' + '--' * 10)
        print(f'{"".join(truth.split(".")[-3:])}')
        print('--' * 10 + 'Fine-tuned Response' + '--' * 10)
        print(f'{"".join(tuned.split(".")[-3:])}')
        print('--' * 10 + 'Base Model Response' + '--' * 10)
        print(f'{"".join(normal.split(".")[-3:])}')
        print('--' * 50)


def save_results(finetuned_outputs, normal_outputs, responses, output_dir="./evaluation_results"):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dictionary with evaluation data
    data = {
        'finetuned_outputs': finetuned_outputs,
        'normal_outputs': normal_outputs,
        'ground_truth_responses': responses
    }
    
    # Save as pickle
    with open(os.path.join(output_dir, 'model_outputs.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Save as JSON for readable format
    with open(os.path.join(output_dir, 'model_outputs.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f'Evaluation results saved to {output_dir}')


def main():
    # Configuration
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    
    # Model paths (modify these according to your setup)
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    finetuned_model_path = "./deepseek_finetuned_model"  # Path to your fine-tuned model
    
    # Check if fine-tuned model exists
    if not os.path.exists(finetuned_model_path):
        print(f"Fine-tuned model not found at {finetuned_model_path}")
        print("Please run the training script first or provide the correct path.")
        return
    
    print("Loading dataset...")
    ds = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B")
    
    print("Creating filtered dataset...")
    filtered_ds = create_dataset(ds)
    
    # Clean up original dataset
    ds.cleanup_cache_files()
    del ds
    
    # Process dataset
    filtered_ds = filtered_ds.map(split_response)
    filtered_ds = filtered_ds.map(formatting_prompts_func, batched=True)
    
    # Split dataset
    _, test_ds, _ = split_dataset(filtered_ds)
    
    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    print("Loading fine-tuned model...")
    finetuned_model, _ = FastLanguageModel.from_pretrained(
        model_name=finetuned_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Set models to inference mode
    FastLanguageModel.for_inference(base_model)
    FastLanguageModel.for_inference(finetuned_model)
    
    print(f"Base model parameters: {base_model.num_parameters():,}")
    print(f"Fine-tuned model parameters: {finetuned_model.num_parameters():,}")
    
    # Filter for hard examples for evaluation
    instructions = test_ds['instruction']
    difficulty = test_ds['difficulty']
    
    hard_ids = [i for i, x in enumerate(difficulty) if x == 'hard']
    print(f"Found {len(hard_ids)} hard examples for evaluation")
    
    hard_instructions = [instructions[i] for i in hard_ids]
    ground_truth_responses = [test_ds['response'][i] for i in hard_ids]
    
    # Define prompt template for inference
    prompt_template = """Below is an instruction that describes a mathematical task, paired with additional context information to guide the solution.
Write a response that thoroughly solves the given problem.
Before solving, develop a clear step-by-step chain of reasoning to ensure accuracy and logical coherence.

### Instruction:
You are a mathematics expert with advanced knowledge in mathematical reasoning, problem-solving, and proof techniques. You think outloud and consider various aspects before giving any concrete answers.

### Question:
{}

### Response:
<think>{}"""
    
    print("Generating responses with fine-tuned model...")
    finetuned_outputs = generate_responses(finetuned_model, tokenizer, hard_instructions, prompt_template)
    
    print("Generating responses with base model...")
    normal_outputs = generate_responses(base_model, tokenizer, hard_instructions, prompt_template)
    
    print(f"Generated {len(normal_outputs)} base model responses and {len(finetuned_outputs)} fine-tuned responses")
    
    # Compare responses
    compare_responses(finetuned_outputs, normal_outputs, ground_truth_responses)
    
    # Save results
    save_results(finetuned_outputs, normal_outputs, ground_truth_responses)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
