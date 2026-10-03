#Load necessary libraries 
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
import torch
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Choose the fine-tuned model you want to evaluate
model_path = str(PROJECT_DIR / "models" / "flan_t5_xray_final")


# Define the constants
INPUT_MAX_LENGTH = 169
OUTPUT_MAX_LENGTH = 119

# Load the fine-tuned model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

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


# Define a list of sample findings to test the model
appendix_samples = [
    "The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.",
    "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities.",
    "AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion."
]


# Generate and print the impressions for the sample findings

print("\n" + "="*40)
print(" APPENDIX SAMPLE PREDICTIONS ")
print("="*40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

output_lines = [
    "=" * 40,
    "APPENDIX SAMPLE PREDICTIONS",
    "=" * 40,
]

for idx, sample in enumerate(appendix_samples, 1):
    predicted_impression = generate_impression(sample)
    output_lines.append(f"\n[Sample {idx}]")
    output_lines.append(f"Input Findings:\n{sample}")
    output_lines.append(f"\nGenerated Impression:\n{predicted_impression}")
    output_lines.append("-" * 40)

    print(f"\n[Sample {idx}]")
    print(f"Input Findings:\n{sample}")
    print(f"\nGenerated Impression:\n{predicted_impression}")
    print("-" * 40)

text_output_path = ARTIFACTS_DIR / "sample_predictions_output.txt"
text_output_path.write_text("\n".join(output_lines), encoding="utf-8")
print(f"\nSaved sample output text to: {text_output_path}")