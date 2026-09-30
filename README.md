# AI Chatbot - Task 3: Fine-Tuning

A fine-tuned chatbot developed using a pretrained FLAN-T5 language model and a custom question-answer dataset.

## Objective

The objective of Task 3 is to fine-tune a pretrained language model using a custom chatbot dataset.

Fine-tuning allows the pretrained model to adapt its response generation according to the examples provided in the training dataset.

## Technologies Used

- Python
- Hugging Face Transformers
- FLAN-T5
- Hugging Face Datasets
- PyTorch
- Pandas
- SentencePiece

## Model Used

The pretrained model used in this project is:

google/flan-t5-small

FLAN-T5 is a text-to-text transformer model that can be adapted to different natural language processing tasks.

## Dataset

A custom question-and-answer dataset is used for fine-tuning.

The dataset contains examples related to:

- Greetings
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Cybersecurity
- Phishing
- Malware
- Encryption
- Ransomware
- General chatbot questions

## Fine-Tuning Process

The system follows these steps:

```text
Custom Dataset
      ↓
Data Preprocessing
      ↓
Tokenization
      ↓
Pretrained FLAN-T5
      ↓
Fine-Tuning
      ↓
Fine-Tuned Model
      ↓
Response Generation
