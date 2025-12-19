#---------------------------------------------------------------------------------------------------------#
# Imports and setup
#---------------------------------------------------------------------------------------------------------#

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#---------------------------------------------------------------------------------------------------------#
# Load dataset
#---------------------------------------------------------------------------------------------------------#

dataset = load_dataset("ag_news")  # Example: AG News, adjust if you use a different one

print(dataset)
# Output should show train/test split and features: ['text', 'label']

#---------------------------------------------------------------------------------------------------------#
# Tokenize the dataset
#---------------------------------------------------------------------------------------------------------#

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# Tokenize first
train_dataset = dataset["train"].map(tokenize, batched=True)
test_dataset  = dataset["test"].map(tokenize, batched=True)

# THEN set torch format (this is the important fix)
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

#---------------------------------------------------------------------------------------------------------#
# Create PyTorch Dataloaders
#---------------------------------------------------------------------------------------------------------#

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

#---------------------------------------------------------------------------------------------------------#
# Load a pre-trained model for sequence classification
#---------------------------------------------------------------------------------------------------------#

num_labels = len(set(dataset['train']['label']))  # auto-detect number of classes

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

model.to(device)

#---------------------------------------------------------------------------------------------------------#
# Define optimizer and loss
#---------------------------------------------------------------------------------------------------------#

# Freeze encoder
for param in model.distilbert.parameters():
    param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

#---------------------------------------------------------------------------------------------------------#
# Training loop (simplified)
#---------------------------------------------------------------------------------------------------------#

epochs = 1  # Start small

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

#---------------------------------------------------------------------------------------------------------#
# Saving the model
#---------------------------------------------------------------------------------------------------------#

output_dir = "./model"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model saved to ./model")
