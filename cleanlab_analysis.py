import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

def main():
    # Load and embed
    data = load_dataset("polyai/banking77")
    train = pd.DataFrame(data["train"])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(train['text'], show_progress_bar=True)

    # Train logistic regression
    clf = LogisticRegressionCV(cv=5, class_weight="balanced", max_iter=1000)
    clf.fit(embeddings, train['label'])
    prob = clf.predict_proba(embeddings)

    # Detect label issues
    issues = find_label_issues(
        labels=train['label'].values,
        pred_probs=prob,
        return_indices_ranked_by="self_confidence"
    )
    print(f"Detected {len(issues)} potential label issues")
    print(train.iloc[issues][['text','label']])

    # Near duplicates
    sim_matrix = cosine_similarity(embeddings)
    threshold = 0.9
    dup_pairs = np.argwhere(np.triu(sim_matrix, k=1) > threshold)
    print(f"Detected {len(dup_pairs)} near duplicate pairs with similarity > {threshold}")
    for i,j in dup_pairs[:10]:
        print(f"{i} vs {j}:")
        print(" -", train.loc[i,'text'])
        print(" -", train.loc[j,'text'])
        print()

    # Outliers
    iso = IsolationForest(contamination=0.01, random_state=42)
    outlier_preds = iso.fit_predict(embeddings)
    outliers = np.where(outlier_preds==-1)[0]
    print(f"Detected {len(outliers)} outliers")
    print(train.iloc[outliers[:10]][['text','label']])

if __name__ == "__main__":
    main()
