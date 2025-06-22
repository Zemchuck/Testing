import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from sklearn.metrics import classification_report, f1_score

def main():
    # Load data and model
    raw = load_dataset("polyai/banking77")
    test = raw['test']
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('./model')
    model.eval()

    # Tokenize and predict
    texts = test['text']
    labels = test['label']
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    # Reports
    print(classification_report(labels, preds, digits=4, zero_division=0))

    # F1 per class plot
    f1s = f1_score(labels, preds, average=None)
    plt.figure(figsize=(12,6))
    plt.bar(range(len(f1s)), f1s)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.tight_layout()
    plt.savefig('f1_per_class.png')
    plt.close()

if __name__ == "__main__":
    main()
