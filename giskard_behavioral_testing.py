import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
from giskard import Dataset, Model, scan


def build_pipeline():
    """Build a Hugging Face pipeline that returns full 77‑class probabilities."""
    model = DistilBertForSequenceClassification.from_pretrained("./model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model.eval()

    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=77,               # return scores for every class
        function_to_apply="softmax"  # ensure probabilities sum to 1
    )


# Load pipeline once (avoids re‑loading inside prediction function)
MODEL_PIPE = build_pipeline()


@torch.no_grad()
def prediction_function(df: pd.DataFrame) -> np.ndarray:
    """Giskard‑compatible prediction function → (n_samples, 77) probs."""
    outputs = MODEL_PIPE(df["text"].tolist())

    probs = np.zeros((len(outputs), 77), dtype=float)
    for i, sample in enumerate(outputs):
        for entry in sample:
            # label may be 'LABEL_32' or '32'
            label_idx = int(entry["label"].split("_")[-1])
            probs[i, label_idx] = entry["score"]

    return probs


def main():
    # --------------------------- Load data ---------------------------
    test_df = pd.DataFrame(load_dataset("polyai/banking77")["test"])

    # ---------------------- Wrap for Giskard ------------------------
    giskard_dataset = Dataset(test_df, target="label", name="banking77_test")
    giskard_model = Model(
        model=prediction_function,
        model_type="classification",
        classification_labels=list(range(77)),
        feature_names=["text"],
        name="banking77_classifier",
    )

    # ------------------------ Run scan ------------------------------
    scan_report = scan(giskard_model, giskard_dataset, verbose=False)
    with open("giskard_behavioural_report.html", "w") as f:
        f.write(scan_report.to_html())

    print("✔ Saved giskard_behavioural_report.html")


if __name__ == "__main__":
    main()
