import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from evaluate import load
import numpy as np

# Step 1: Prepare the Dataset
# Create a synthetic QA dataset from the provided knowledge base excerpts
data = [
    {
        "question": "Who is Dr Muhammad Ather Khurram?",
        "answer": "Dr Muhammad Ather Khurram is the Chief Executive Officer of Media Network International and Chief Operating Officer of all operations, projects, and TV channels of Media Network International in Pakistan. He is a famous Pakistani journalist, columnist, and political analyst born on April 22, 1977."
    },
    {
        "question": "What are Dr Muhammad Ather Khurram's educational qualifications?",
        "answer": "He holds a PhD in Mass Communication and is a Gold Medalist in Master of Computer Sciences and Master of Business Administration and Information Technology."
    },
    {
        "question": "What newspapers has Dr Muhammad Ather Khurram edited?",
        "answer": "He has served as Editor or Chief Editor for newspapers including Daily Taqat, Daily Masafat, Daily Asas, Daily Lashkar, Daily Times of Pakistan, Daily Jang, Daily Samundri News, Daily Muhib e Watan, Daily Sadaqat, Daily Inqilab, Daily Pakistan, and Express News."
    },
    {
        "question": "What books has Dr Muhammad Ather Khurram authored?",
        "answer": "He authored 'Azaan Kay Baad' published in 1998 and 'Haqeeqat E Mohabbat' published in 2000."
    },
    {
        "question": "What leadership roles does Dr Muhammad Ather Khurram hold?",
        "answer": "He is chairman of Think Tank of Pakistan, chairman of Pakistan Institute of Media Sciences, President of Institute of Modern Journalism, and General Secretary of Institute of National Affairs, Institute of Media & Politics, Institute of Media and Society, and Pakistan Media Forum."
    },
    {
        "question": "Who was Thomas Thomson?",
        "answer": "Thomas Thomson (4 December 1817 – 18 April 1878) was a Scottish surgeon with the British East India Company who later became a botanist. He helped write the first volume of Flora Indica and was a friend of Joseph Dalton Hooker."
    },
    {
        "question": "What was Thomas Thomson's educational background?",
        "answer": "He qualified as an M.D. at Glasgow University in 1839."
    },
    {
        "question": "What military campaigns did Thomas Thomson participate in?",
        "answer": "He served in the Afghanistan campaign (1839-1842), the Sutlej campaign (1845-46), and the second Sikh war (1848-49)."
    },
    {
        "question": "What roles did Thomas Thomson hold in botany?",
        "answer": "He was Superintendent of the Honourable East India Company's Botanic Garden at Calcutta and Naturalist to and Member of the Tibet Mission."
    },
    {
        "question": "What honors did Thomas Thomson receive?",
        "answer": "He was appointed a Fellow of the Royal Society in 1855 and received the Royal Geographical Society's Founder's Medal in 1866."
    },
]

# Convert to a Hugging Face Dataset
dataset = Dataset.from_list([
    {"text": f"Question: {item['question']}\nAnswer: {item['answer']}", "label": 1} for item in data
])
# Split into train (80%) and validation (20%)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# Step 2: Load Tokenizer and Model
model_name = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)  # Binary classification for simplicity

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 3: Configure LoRA for PEFT
lora_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["query", "value"],  # BERT attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Check the number of trainable parameters

# Step 4: Define Compute Metrics
accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Step 5: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_bert_knowledge_base",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 6: Initialize Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_bert_knowledge_base")
tokenizer.save_pretrained("./fine_tuned_bert_knowledge_base")

# Step 9: Example Inference
def predict(question):
    inputs = tokenizer(
        f"Question: {question}\nAnswer:", padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Valid answer" if predicted_class == 1 else "Invalid answer"

# Test the model
test_question = "What is Dr Muhammad Ather Khurram's role at Media Network International?"
print(predict(test_question))
