import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds, average='macro')}


def main():
    # Load Banking77 dataset
    raw = load_dataset("polyai/banking77")

    # Train-validation split (80/20)
    split = raw['train'].train_test_split(test_size=0.2, seed=42)
    datasets = DatasetDict({
        'train': split['train'],
        'validation': split['test'],
        'test': raw['test']
    })

    # Initialize tokenizer and collator
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    data_collator = DataCollatorWithPadding(tokenizer)

    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True)

    # Apply tokenization
    tokenized = datasets.map(tokenize, batched=True)

    # Load pre-trained DistilBERT for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=77
    )

    # Training arguments (without evaluation_strategy for compatibility)
    training_args = TrainingArguments(
        output_dir='./results',
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Initialize Trainer with data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and save model
    trainer.train()
    trainer.save_model('./model')


if __name__ == "__main__":
    main()
