from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load GPT-2 tokenizer and model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load your custom dataset (make sure it's in the right format)
dataset = load_dataset("text", data_files={"train": "kid_friendly_stories.txt"}, split="train")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=500,
    do_train=True,
    do_eval=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You can create a separate validation set if needed
)

# Train the model
trainer.train()
