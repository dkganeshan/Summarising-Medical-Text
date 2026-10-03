from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent

model_path = str(PROJECT_DIR / "models" / "flan_t5_xray_final")

app = FastAPI(title="Radiology Findings Summarizer")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load model and tokenizer with proper inference config
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

# Move to CPU (avoid GPU memory issues) or GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class FindingsRequest(BaseModel):
    findings: str

class ImpressionResponse(BaseModel):
    impression: str

@app.post("/summarize", response_model=ImpressionResponse)
def summarize(request: FindingsRequest):
    findings = request.findings.strip()
    if not findings:
        raise HTTPException(status_code=400, detail="Missing 'findings' field in request body")

    # Tokenize and move inputs to same device as model
    inputs = tokenizer("Summarize the following X-ray findings: " + findings, return_tensors="pt", truncation=True, max_length=256)  
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=64            
        )
        impression = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"impression": impression}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
