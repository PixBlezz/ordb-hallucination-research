"""
Step 1: Binary-judgment synthesis pipeline.
Recreates the donation platform's actual structure:
  - Each item gets a binary label (acceptable / not acceptable)
  - Then those labels (not the raw text) get synthesized into one review

This is DIFFERENT from the free-text pipeline we already tested.
There, the model saw full review text. Here, it sees only labels --
much less information to work with, closer to what actually happened
on the donation platform.
"""

import pandas as pd
import json

def score_to_label(score: int) -> str:
    """Mirrors the donation platform's acceptable/not-acceptable judgment."""
    if score >= 4:
        return "acceptable"
    elif score <= 2:
        return "not acceptable"
    else:
        return "borderline"  # score == 3

def build_binary_judgment_prompt(product_id: str, labels: list) -> str:
    label_lines = "\n".join(f"- Item {i+1}: {lbl}" for i, lbl in enumerate(labels))
    n = len(labels)
    n_accept = labels.count("acceptable")
    n_reject = labels.count("not acceptable")
    n_border = labels.count("borderline")

    prompt = (
        f"You are reviewing a batch of {n} items for product {product_id}.\n"
        f"Each item has already been individually checked and labeled:\n\n"
        f"{label_lines}\n\n"
        "Write one overall review summarizing this batch for a customer, "
        "based only on the labels above. Do not invent details about specific "
        "items beyond what the labels tell you."
    )
    return prompt, {"n": n, "n_accept": n_accept, "n_reject": n_reject, "n_border": n_border}


if __name__ == "__main__":
    df = pd.read_csv("data/Reviews.csv")

    # Use the SAME 12 real product groups from the earlier free-text batch,
    # so this is a clean, matched comparison -- same data, different task.
    batch = pd.read_csv("data/batch_selection.csv")

    all_prompts = {}
    for _, row in batch.iterrows():
        pid = row["ProductId"]
        scores = df[df["ProductId"] == pid]["Score"].tolist()
        labels = [score_to_label(s) for s in scores]
        prompt, stats = build_binary_judgment_prompt(pid, labels)
        all_prompts[pid] = {"prompt": prompt, "stats": stats, "labels": labels}

        with open(f"data/binary_prompt_{pid}.txt", "w") as f:
            f.write(prompt)

    with open("data/binary_prompts_all.json", "w") as f:
        json.dump(all_prompts, f, indent=2)

    print(f"Built binary-judgment prompts for {len(all_prompts)} product groups.\n")
    for pid, d in all_prompts.items():
        print(f"{pid}: {d['stats']['n']} items -> "
              f"{d['stats']['n_accept']} acceptable, "
              f"{d['stats']['n_reject']} not acceptable, "
              f"{d['stats']['n_border']} borderline")

    print("\n--- Example prompt (first product) ---\n")
    first_key = list(all_prompts.keys())[0]
    print(all_prompts[first_key]["prompt"])
