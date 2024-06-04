# llm-finetune-financial-advisor

## Use Case
The project aims to fine-tune the Llama 2 model for financial advisory tasks. By customizing the model with specific financial data, the model can provide detailed and accurate financial advice based on user queries.

## Code Steps

### Step 1: Install Required Packages
Ensure all necessary packages are installed:
```shell
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 datasets evaluate requests==2.31.0
```

### Step 2: Import Required Libraries
Import the essential libraries such as PyTorch, Hugging Face transformers, datasets, and others needed for fine-tuning.

### Step 3: Reformat the Instruction Dataset
Reformat the dataset to match the Llama 2 template for instructions, ensuring the data includes clear and detailed guidelines for providing answers to financial queries.

### Step 4: Fine-tune Llama 2
Define the model and dataset for fine-tuning:
- **Model**: `togethercomputer/Llama-2-7B-32K-Instruct`
- **Dataset**: `nihiluis/financial-advisor-100`

Configure QLoRA parameters and bitsandbytes for 4-bit quantization. Load the model and tokenizer with these configurations.

### Step 5: Load and Process Dataset
Load and preprocess the dataset, ensuring it's formatted correctly and ready for training.

### Step 6: Configure and Start Fine-Tuning
Set training parameters and initialize the `SFTTrainer` with these configurations. Start the fine-tuning process, monitoring training loss and performance metrics.

### Step 7: Save the Trained Model
Save the fine-tuned model for future use.

### Step 8: Monitor Training with TensorBoard
Use TensorBoard to visualize and monitor the training process.

### Step 9: Use the Model for Text Generation
After fine-tuning, use the model to generate financial advice based on user queries, formatted to match the Llama 2 prompt template.

## Conclusion
This project demonstrates the steps required to fine-tune a large language model (LLM) like Llama 2 for a specific use case—providing financial advice. By customizing the model with domain-specific data and following a structured fine-tuning process, the model can deliver accurate and detailed responses tailored to financial advisory contexts.
