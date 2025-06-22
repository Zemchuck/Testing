"""captum_explainability.py
--------------------------------------------------
Token‑level explanations for a fine‑tuned **DistilBERT Banking77** classifier
using **Layer Integrated Gradients** (Captum).

Example:
    uv run python captum_explainability.py \
        --text "I want to open a new savings account" \
        --target 4 \
        --model-path ./model
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
from captum.attr import (
    LayerIntegratedGradients,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization as viz,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
PAD_ID: int = TOKENIZER.pad_token_id


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """Load a fine‑tuned checkpoint and move it to CPU / GPU."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
    model = DistilBertForSequenceClassification.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def _strip_special(tokens, attributions):
    """Drop [CLS], [SEP], [PAD] tokens for cleaner display."""
    show_tok, show_attr = [], []
    for tok, att in zip(tokens, attributions):
        if tok not in {"[CLS]", "[SEP]", "[PAD]"}:
            show_tok.append(tok)
            show_attr.append(float(att))
    return show_tok, show_attr


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def explain_text(text: str, target_label: int, model_path: str) -> None:
    """Visualise token attributions for *text* using Layer Integrated Gradients."""

    # 1 ▸ model -----------------------------------------------------------
    model, device = _load_model(model_path)

    # 2 ▸ tokenise --------------------------------------------------------
    enc = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 3 ▸ interpretable embedding layer (ONLY word embeddings) ------------
    ie_layer = configure_interpretable_embedding_layer(
        model, "distilbert.embeddings.word_embeddings"
    )

    # 4 ▸ forward function (standard IDs) --------------------------------
    def forward_fn(ids, attn_mask):
        return model(input_ids=ids, attention_mask=attn_mask).logits

    # 5 ▸ Layer Integrated Gradients -------------------------------------
    lig = LayerIntegratedGradients(forward_fn, ie_layer)

    baseline_ids = torch.full_like(input_ids, PAD_ID)

    try:
        attributions, _ = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True,
        )
    finally:
        # Always restore original embeddings even if IG fails
        remove_interpretable_embedding_layer(model, ie_layer)

    # 6 ▸ post‑process ----------------------------------------------------
    token_attr = attributions.sum(dim=-1).squeeze(0)
    token_attr = token_attr / (token_attr.norm() + 1e-8)

    tokens = TOKENIZER.convert_ids_to_tokens(input_ids[0])
    disp_tok, disp_attr = _strip_special(tokens, token_attr)

    # 7 ▸ model prediction (for header) ----------------------------------
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)[0]
        pred_cls = int(probs.argmax())
        pred_prob = float(probs[pred_cls])

    # 8 ▸ visualise -------------------------------------------------------
    record = viz.VisualizationDataRecord(
        word_attributions=disp_attr,
        pred_prob=pred_prob,
        pred_class=pred_cls,
        true_class=target_label,
        attr_class=target_label,
        attr_score=sum(disp_attr),
        raw_input_ids=disp_tok,
        convergence_score=0.0,
    )
    viz.visualize_text([record])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token‑level explanations with Captum + DistilBERT",
    )
    parser.add_argument("--text", required=True, help="Input text to explain")
    parser.add_argument("--target", type=int, default=0, help="Label index (0‑76)")
    parser.add_argument(
        "--model-path",
        default="./model",
        help="Directory with fine‑tuned DistilBERT checkpoint",
    )

    args = parser.parse_args()
    explain_text(args.text, args.target, args.model_path)
