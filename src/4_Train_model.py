# Import necessary libraries
from pathlib import Path
from datasets import load_dataset # Hugging Face Datasets library to be installed in the environment
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq, 
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer) # Hugging Face Transformers library to be installed in the environment
import torch # PyTorch to be installed in the environment



PROJECT_DIR = Path(__file__).resolve().parent.parent

# Choose the model you want to fine-tune
model_id = "google/flan-t5-small"


# Define the constants

PROMPT_PREFIX = "Summarize the following X-ray findings: "
INPUT_MAX_LENGTH = 169
OUTPUT_MAX_LENGTH = 119


# Load the tokenizer and model 
    
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


# Define the data collator
# Automatically pads inputs and labels to the longest sequence in each batch
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model
)


# Define the training arguments

training_args = Seq2SeqTrainingArguments(
    output_dir=str(PROJECT_DIR / "models" / "flan_t5_xray_summarizer"),
    per_device_train_batch_size=4,   # Keep small for CPU memory efficiency
    per_device_eval_batch_size=4,
    predict_with_generate=True,      # Needed for Seq2Seq generation evaluation
    learning_rate=5e-4,              # Slightly higher learning rate for smaller models
    num_train_epochs=3,              # 3-5 epochs is typical for fine-tuning
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    use_cpu=not torch.cuda.is_available(),
    fp16=torch.cuda.is_available(),   # Use mixed precision on supported CUDA GPUs
    report_to="none"                 # Disables external tracking logins (like WandB)
)



# Define the function to apply the prompt template to each row of the dataset
def apply_prompt_template(example):
    # The instruction prefix we discussed
    prefix = PROMPT_PREFIX
    
    # Combine the prefix with the raw findings text
    full_input = prefix + example["findings"]
    
    # Grab the target text (Impression) from the dataset
    target = example["impression"]
    
    # Return a dictionary with our new, formatted columns
    return {
        "text_input": full_input, 
        "text_target": target
    }



# Define the tokenization function

def preprocess_function(examples):
    # 1. Tokenize input findings
    model_inputs = tokenizer(
        examples["text_input"], 
        max_length=INPUT_MAX_LENGTH, 
        truncation=True
    )
    
    # 2. Tokenize target impressions
    labels = tokenizer(
        text_target=examples["text_target"], 
        max_length=OUTPUT_MAX_LENGTH, 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



#.......................................................................................


# 1. Load your CSV file into a Hugging Face Dataset

train_csv = PROJECT_DIR / "data" / "processed_data" / "train_data.csv"

if not train_csv.exists():
    raise FileNotFoundError(f"Training CSV not found: {train_csv}")

train_dataset = load_dataset("csv", data_files=str(train_csv))



# 2. Apply the prompt template to the dataset
formatted_train_dataset = train_dataset.map(apply_prompt_template)
print("check point 1: Formatted datasets created successfully.")


# 3. Tokenize the dataset using the preprocess function

tokenized_train = formatted_train_dataset.map(preprocess_function, batched=True)
print("check point 2: Tokenization completed successfully.")



# 4. Start fine-tuning

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train["train"],
    eval_dataset=tokenized_train["train"],
    processing_class=tokenizer,
    data_collator=data_collator,
)
trainer.train()



# 5. Save model and tokenizer for inference
trainer.save_model(str(PROJECT_DIR / "models" / "flan_t5_xray_final"))
tokenizer.save_pretrained(str(PROJECT_DIR / "models" / "flan_t5_xray_final"))
print("Model and tokenizer saved successfully.")