# =========================================================
# TASK 3 - FINE-TUNED CHATBOT
# =========================================================

import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)


# =========================================================
# 1. LOAD CUSTOM DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("dataset.csv")

# Remove empty rows
df = df.dropna()

print("Number of training examples:", len(df))

dataset = Dataset.from_pandas(
    df[["input", "response"]],
    preserve_index=False
)


# =========================================================
# 2. LOAD PRETRAINED FLAN-T5 MODEL
# =========================================================

model_name = "google/flan-t5-small"

print()
print("Loading pretrained model:", model_name)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)


# =========================================================
# 3. TOKENIZE DATASET
# =========================================================

def preprocess_function(examples):

    model_inputs = tokenizer(
        examples["input"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["response"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print()
print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)


# =========================================================
# 4. TRAINING CONFIGURATION
# =========================================================

training_args = TrainingArguments(
    output_dir="./fine_tuned_chatbot",

    num_train_epochs=3,

    per_device_train_batch_size=4,

    learning_rate=2e-5,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none",

    fp16=torch.cuda.is_available()
)


# =========================================================
# 5. CREATE TRAINER
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)


# =========================================================
# 6. FINE-TUNE MODEL
# =========================================================

print()
print("=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)

trainer.train()


# =========================================================
# 7. SAVE FINE-TUNED MODEL
# =========================================================

print()
print("Saving fine-tuned model...")

trainer.save_model(
    "./fine_tuned_chatbot"
)

tokenizer.save_pretrained(
    "./fine_tuned_chatbot"
)

print("Model saved successfully!")


# =========================================================
# 8. LOAD FINE-TUNED MODEL
# =========================================================

print()
print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(
    "./fine_tuned_chatbot"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "./fine_tuned_chatbot"
)

model.eval()


# =========================================================
# 9. CHATBOT FUNCTION
# =========================================================

def chatbot(user_input):

    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# =========================================================
# 10. CHAT INTERFACE
# =========================================================

print()
print("=" * 60)
print("FINE-TUNED CHATBOT")
print("=" * 60)

print("Ask a question to the chatbot.")
print("Type 'bye' to exit.")
print()


while True:

    user_input = input("You: ").strip()

    if user_input.lower() in [
        "bye",
        "goodbye",
        "exit",
        "quit"
    ]:

        print("Bot: Goodbye!")
        break

    if not user_input:

        print("Bot: Please enter a question.")
        continue

    response = chatbot(
        user_input
    )

    print("Bot:", response)
    print()
