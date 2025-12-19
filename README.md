# Multi-Class Text Classification API
This project is a A Python-based API for classifying news articles into one of four categories: World, Sports, Business, Sci/Tech. Built with FastAPI, PyTorch, and Hugging Face Transformers.

------------------------------------------------------------------------------------------------------
## Overview
------------------------------------------------------------------------------------------------------

This project demonstrates:

Fine-tuning a pre-trained DistilBERT model for text classification on the AG News dataset.
Building a FastAPI backend for serving the model.
Creating a REST API endpoint /predict that returns predicted class and confidence for any input text.
Packaging the model and API for easy local use.

The API allows users to send any text and get a classification into one of the four categories.

------------------------------------------------------------------------------------------------------
## Features
------------------------------------------------------------------------------------------------------

-Fast, lightweight model using DistilBERT.
-Easy-to-use API via /predict endpoint.
-Local evaluation script to measure accuracy on the test set.
-Error handling for empty or invalid inputs.
-Supports CPU and GPU inference.

------------------------------------------------------------------------------------------------------
## Files in the Repository
------------------------------------------------------------------------------------------------------

train.py ------------------ Fine-tune the DistilBERT model on AG News.
evaluate.py ------------------ Evaluate the model accuracy on the test set.
app.py ------------------ FastAPI server for serving predictions.
model/ ------------------ Saved pre-trained model and tokenizer files.
requirements.txt ------------------ Python dependencies.
README.md ------------------ Project overview and instructions.

------------------------------------------------------------------------------------------------------
## Prerequisites
------------------------------------------------------------------------------------------------------

Python 3.11
pip
Optional: GPU for faster inference

------------------------------------------------------------------------------------------------------
## Setup Instructions
------------------------------------------------------------------------------------------------------

1. Clone the repository

git clone <your-repo-url>
cd <repo-folder>

2. Create a virtual environment

python -m venv .venv

3. Activate the virtual environment
   
   Windows:
   ".venv\Scripts\activate"

    macOS/Linux:
    "source .venv/bin/activate"

4. Install dependencies

"pip install -r requirements.txt"

5. Ensure the model/ folder exists

The API requires a trained model which isn't included. Run:

"python train.py"

Then evaluate the model but running: 

"python evaluate.py"

This is to test the model on the AG News test set.
This ensures the trained model performs as expected.

Output will display the test accuracy. Example:

"Test accuracy: 0.8954"

------------------------------------------------------------------------------------------------------
## Running the API
------------------------------------------------------------------------------------------------------

1. Start the FastAPI server:

uvicorn app:app --reload

2. Open "http://127.0.0.1:8000/docs" to interact with the API using the Swagger UI.

Go to the POST /predict endpoint.

Click Try it out.

Enter any text in the "text" field.

Click Execute to see the predicted class and confidence.

Example request body:

{
  "text": "Apple announces new iPhone 15 release date"
}

Example response:

{
  "predicted_class": "Sci/Tech",
  "confidence": 0.9504
}

------------------------------------------------------------------------------------------------------
## Notes
------------------------------------------------------------------------------------------------------

CPU vs GPU: The model runs on CPU if GPU is unavailable, but training or inference may be slower.

Max token length: Input text is truncated to 64 tokens for speed and memory efficiency.

Freezing DistilBERT: During training, the encoder is frozen to reduce memory usage.