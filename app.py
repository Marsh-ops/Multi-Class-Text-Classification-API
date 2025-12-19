from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_dir = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI setup
app = FastAPI(title="Multi-Class Text Classification API")

# Input schema with custom placeholder
class TextIn(BaseModel):
    text: str = Field(
        ...,
        example="Type your news text here (e.g., 'Apple releases new iPhone 15')"
    )

# Class labels
class_labels = ["World", "Sports", "Business", "Sci/Tech"]

# Prediction endpoint with error handling
@app.post("/predict")
def predict(data: TextIn):
    # Error handling: empty string
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # Error handling: too long input
    if len(data.text) > 500:  # you can adjust the max length
        raise HTTPException(status_code=400, detail="Text input is too long. Limit to 500 characters.")

    # Tokenize
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_index].item()

    return {
        "predicted_class": class_labels[pred_index],
        "confidence": round(confidence, 4)
    }

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome! Use /predict to get predictions on your news text."}