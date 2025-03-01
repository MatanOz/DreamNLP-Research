import os
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset

# Define paths
TRAIN_TXT_PATH = "data/train_dreams.txt"
VAL_TXT_PATH = "data/val_dreams.txt"
MODEL_DIR = "models/fine_tuned_gpt2"
LAST_CHECKPOINT = None

# Check for previous checkpoints
if os.path.exists(MODEL_DIR) and any(fname.startswith("checkpoint") for fname in os.listdir(MODEL_DIR)):
    checkpoints = [f for f in os.listdir(MODEL_DIR) if f.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))  # Sort by step number
    LAST_CHECKPOINT = os.path.join(MODEL_DIR, checkpoints[-1])
    print(f"✅ Resuming training from checkpoint: {LAST_CHECKPOINT}")

# Ensure datasets exist
if not os.path.exists(TRAIN_TXT_PATH) or not os.path.exists(VAL_TXT_PATH):
    raise FileNotFoundError(f"🚨 Processed dataset not found! Run `python scripts/data_prep.py` first.")

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Fix padding issue
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

model = AutoModelForCausalLM.from_pretrained("gpt2" if LAST_CHECKPOINT is None else LAST_CHECKPOINT)

# Function to load and tokenize dataset with labels
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read().split("\n")
    
    def tokenize_function(examples):
        encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        encodings["labels"] = encodings["input_ids"].copy()  # Labels should match input IDs for GPT-2 training
        return encodings

    dataset = Dataset.from_dict({"text": text_data}).map(tokenize_function, batched=True)
    return dataset

# Load train and validation datasets
train_dataset = load_dataset(TRAIN_TXT_PATH)
val_dataset = load_dataset(VAL_TXT_PATH)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy="epoch",
    evaluation_strategy="epoch",  # Now we evaluate after each epoch
    logging_dir="logs",
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    resume_from_checkpoint=LAST_CHECKPOINT if LAST_CHECKPOINT else None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Now we have a validation dataset
    tokenizer=tokenizer
)

if __name__ == "__main__":
    print("🚀 Starting GPT-2 fine-tuning...")
    trainer.train(resume_from_checkpoint=LAST_CHECKPOINT)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Training complete! Model saved to '{MODEL_DIR}'.")
