#Load necessary libraries 
import csv
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq)
from evaluate import load # Hugging Face Evaluate library to be installed in the environment
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Choose the fine-tuned model you want to evaluate
model_path = str(PROJECT_DIR / "models" / "flan_t5_xray_final")


# Define the constants
PROMPT_PREFIX = "Summarize the following X-ray findings: "
INPUT_MAX_LENGTH = 169
OUTPUT_MAX_LENGTH = 119

# Load the fine-tuned model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


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



# Define a function to generate impressions from findings using the fine-tuned model
def generate_impression(findings):
    prompt = "Summarize the following X-ray findings: " + findings
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=INPUT_MAX_LENGTH
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=OUTPUT_MAX_LENGTH
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)



# 1. Load your CSV file into a Hugging Face Dataset


test_dataset = load_dataset("csv", data_files={"test": str(PROJECT_DIR / "data" / "processed_data" / "test_data.csv")})




# 2. Apply the prompt template to the dataset  

formatted_test_dataset = test_dataset.map(apply_prompt_template)

print("check point 1: Formatted dataset created successfully.")



# 4. Generate predictions for the test dataset

preds = []

# Access the test split explicitly

test_split = formatted_test_dataset["test"]

for text in test_split["findings"]:
    prompt = (
        PROMPT_PREFIX
        + text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=INPUT_MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=OUTPUT_MAX_LENGTH
        )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    preds.append(prediction)



# 5. Get reference impressions

references = test_split["impression"]

# Save paired prediction vs actual impression in JSON and CSV
paired_results = [
    {
        "index": idx,
        "findings": text,
        "actual_impression": ref,
        "predicted_impression": pred
    }
    for idx, (text, ref, pred) in enumerate(zip(test_split["findings"], references, preds))
]

json_path = ARTIFACTS_DIR / "test_predictions_vs_actual.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(paired_results, f, ensure_ascii=False, indent=2)

csv_path = ARTIFACTS_DIR / "test_predictions_vs_actual.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["index", "findings", "actual_impression", "predicted_impression"]
    )
    writer.writeheader()
    for row in paired_results:
        writer.writerow(row)

print(f"Saved JSON artifact: {json_path}")
print(f"Saved CSV artifact: {csv_path}")


# 6. Evaluate the predictions using ROUGE

rouge = load("rouge")

f1_score = rouge.compute(
predictions=preds,
references=references,
use_stemmer=True
)

latest_f1 = float(f1_score.get("rouge1", 0.0))
model_details = {
    "model_name": "google/flan-t5-small",
    "local_model_path": model_path,
    "last_trained_timestamp": "unknown",
    "evaluation_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z"
}

model_meta_path = PROJECT_DIR / "models" / "flan_t5_xray_final" / "training_metadata.json"
if model_meta_path.exists():
    with model_meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
        model_details["last_trained_timestamp"] = metadata.get("last_trained_timestamp", "unknown")

metrics_summary = {
    "model": model_details,
    "latest_f1_score": latest_f1,
    "threshold": 0.30,
    "passed": latest_f1 >= 0.30,
    "evaluation_timestamp": model_details["evaluation_timestamp"]
}

metrics_path = ARTIFACTS_DIR / "latest_evaluation_metrics.json"
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

print("ROUGE-1 F1 Score:", latest_f1)
print("PASS" if latest_f1 >= 0.30 else "FAIL - tune and retrain")
print(f"Saved metrics artifact: {metrics_path}")

