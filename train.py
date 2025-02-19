from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
dataset = load_dataset("json", data_files="personal_data.jsonl", split="train")

local_model_path = "./phi-2_local"

# Define model name
MODEL_NAME = "./phi-2_local"

# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",         # Use "nf4" quantization type
    bnb_4bit_use_double_quant=True     # Enable double quantization
)

# Load tokenizer and model (include quantization_config)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config,   # <-- added quantization configuration
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Optional: Verify parameter names to adjust target_modules if needed
# for name, _ in model.named_parameters():
#     print(name)

# ✅ Apply LoRA fine-tuning
peft_config = LoraConfig(
    r=16,  # Increase adaptation power
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj"],  # Adjust if model parameter names differ
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# Check trainable parameters ✅ (should print LoRA adapter parameters)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(example):
    encoding = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    encoding["labels"] = encoding["input_ids"].copy()  # Add labels for loss computation
    return encoding

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=25,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    eval_strategy="no",
    fp16=True,
    optim="adamw_torch",
    dataloader_num_workers=0,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")