import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

def main():
    # Load dataset
    data = load_dataset("polyai/banking77")
    train = pd.DataFrame(data["train"])
    test = pd.DataFrame(data["test"])

    # Data types and counts
    print("Train dtypes:\n", train.dtypes)
    print("Number of training examples:", len(train))

    # Class distribution
    class_counts = train['label'].value_counts().sort_index()
    plt.figure(figsize=(12,6))
    class_counts.plot.bar()
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

    # Text length histogram (in words)
    train['text_length'] = train['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10,5))
    train['text_length'].hist(bins=50)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("text_length_histogram.png")
    plt.close()

if __name__ == "__main__":
    main()
