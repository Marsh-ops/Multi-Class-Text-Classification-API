# Multi-Class Text Classification API
This project is a A Python-based API for classifying news articles into one of four categories: World, Sports, Business, Sci/Tech. Built with FastAPI, PyTorch, and Hugging Face Transformers.

------------------------------------------------------------------------------------------------------
## Overview
------------------------------------------------------------------------------------------------------

This project demonstrates:

1) Fine-tuning a pre-trained DistilBERT model for text classification on the AG News dataset.
2) Building a FastAPI backend for serving the model.
3) Creating a REST API endpoint /predict that returns predicted class and confidence for any input text.
4) Packaging the model and API for easy local use.

The API allows users to send any text and get a classification into one of the four categories.

------------------------------------------------------------------------------------------------------
## Features
------------------------------------------------------------------------------------------------------

1) Fast, lightweight model using DistilBERT.
2) Easy-to-use API via /predict endpoint.
3) Local evaluation script to measure accuracy on the test set.
4) Error handling for empty or invalid inputs.
5) Supports CPU and GPU inference.

------------------------------------------------------------------------------------------------------
## Files in the Repository
------------------------------------------------------------------------------------------------------

1) train.py ------------------ Fine-tune the DistilBERT model on AG News.
2) evaluate.py ------------------ Evaluate the model accuracy on the test set.
3) app.py ------------------ FastAPI server for serving predictions.
4) model/ ------------------ Saved pre-trained model and tokenizer files.
5) requirements.txt ------------------ Python dependencies.
6) README.md ------------------ Project overview and instructions.

------------------------------------------------------------------------------------------------------
## Prerequisites
------------------------------------------------------------------------------------------------------

1) Python 3.11
2) pip
3) Optional: GPU for faster inference

------------------------------------------------------------------------------------------------------
## Setup Instructions
------------------------------------------------------------------------------------------------------

Clone the repository

1) git clone <your-repo-url>
2) cd <repo-folder>

Create a virtual environment

1) "python -m venv .venv"

Activate the virtual environment
   
   1) Windows:
   2) ".venv\Scripts\activate"

    1) macOS/Linux:
    2) "source .venv/bin/activate"

Install dependencies

1) "pip install -r requirements.txt"

Ensure the model/ folder exists

The API requires a trained model which isn't included. Run:

1) "python train.py"

Then evaluate the model but running: 

2) "python evaluate.py"

This is to test the model on the AG News test set.
This ensures the trained model performs as expected.

Output will display the test accuracy. Example:

"Test accuracy: 0.8954"

------------------------------------------------------------------------------------------------------
## Running the API
------------------------------------------------------------------------------------------------------

Start the FastAPI server:

1) "uvicorn app:app --reload"

Open "http://127.0.0.1:8000/docs" to interact with the API using the Swagger UI.

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