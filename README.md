# DeepSeek Math Reasoning Enhancement

This project focuses on fine-tuning the DeepSeek-R1-Distill-Qwen-1.5B model to improve its mathematical reasoning capabilities. By leveraging the Magpie-Reasoning dataset and implementing LoRA (Low-Rank Adaptation), we've enhanced the model's ability to provide clear, concise, and accurate mathematical solutions.

## Project Overview

The project addresses a common issue in language models where they can get stuck in recursive thinking patterns during mathematical problem-solving. Our fine-tuned model demonstrates improved capability in providing direct, accurate answers without falling into destructive loops.

### Key Improvements
- **Original Model Behavior**: Tendency to second-guess calculations with phrases like "Wait, 4 +16 is 20, plus another 16 is 32? Wait, no, 4 +16 is 10..."
- **Fine-tuned Model**: Provides clear, structured responses with definitive answers

## Technical Implementation

### Dataset
- Source: Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B
- Filtering:
  - Selected 'excellent' quality inputs
  - Focused on Math category
  - Required complete responses with thinking steps
  - Maintained stratified distribution across difficulty levels

### Model Architecture
- Base Model: DeepSeek-R1-Distill-Qwen-1.5B
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Key Parameters:
  - LoRA rank: 8
  - LoRA alpha: 8
  - Target modules: Query, Key, Value projections, and FFN components
  - Max sequence length: 4096 tokens

### Training Configuration
- Batch size: 2
- Gradient accumulation steps: 8
- Learning rate: 2e-4
- Training epochs: 3
- Warmup ratio: 0.1
- Optimizer: AdamW (8-bit)
- Early stopping with patience of 2

## Installation and Usage

```bash
# Install required packages
pip install -q unsloth
pip install -q peft transformers trl accelerate bitsandbytes
pip install -q -U datasets
```

### Loading the Fine-tuned Model

```python
from unsloth import FastLanguageModel

# Load the model with LoRA weights
model, tokenizer = FastLanguageModel.from_pretrained(
    "path_to_saved_model",
    max_seq_length=4096,
    load_in_4bit=True
)
```

## Results and Evaluation

Initial evaluations show significant improvements in the model's mathematical reasoning:
- Eliminated recursive doubt patterns ("Wait, no...")
- More concise and confident answers
- Proper mathematical notation using LaTeX formatting
- Clearer step-by-step reasoning

## Future Work

- Comprehensive evaluation across different mathematical domains
- Extension to more complex mathematical problems
- Integration with other mathematical tools and libraries
- Performance optimization for faster inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Magpie-Align for the dataset
- Unsloth team for the optimization library
- DeepSeek AI for the base model
