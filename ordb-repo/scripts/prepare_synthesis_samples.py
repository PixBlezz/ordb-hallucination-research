import pandas as pd
import json

df = pd.read_csv("data/Reviews.csv")

# Pick real sample groups across our three conditions:
# 1. High ambiguity (high score variance) - predicted highest hallucination risk
# 2. Low ambiguity (low score variance) - predicted lowest hallucination risk
# 3. Varying volume (5 vs 10 vs 20) within similar ambiguity, to isolate volume effect

samples = {}

# High ambiguity sample
high_var_id = "B0026LH22U"
g = df[df["ProductId"] == high_var_id][["Score", "Summary", "Text"]]
samples["high_ambiguity_10reviews"] = {
    "ProductId": high_var_id,
    "n_reviews": len(g),
    "avg_score": round(g["Score"].mean(), 2),
    "score_std": round(g["Score"].std(), 2),
    "reviews": g.to_dict("records"),
}

# Low ambiguity, similar size, for contrast
grouped = df.groupby("ProductId").agg(n=("Score", "count"), std=("Score", "std")).reset_index()
low_var_candidates = grouped[(grouped["n"] >= 9) & (grouped["n"] <= 11)].sort_values("std").head(1)
low_var_id = low_var_candidates.iloc[0]["ProductId"]
g2 = df[df["ProductId"] == low_var_id][["Score", "Summary", "Text"]]
samples["low_ambiguity_similar_size"] = {
    "ProductId": low_var_id,
    "n_reviews": len(g2),
    "avg_score": round(g2["Score"].mean(), 2),
    "score_std": round(g2["Score"].std(), 2),
    "reviews": g2.to_dict("records"),
}

def build_prompt(sample):
    lines = "\n".join(f"- (Rating {r['Score']}/5) {r['Summary']}: {r['Text']}" for r in sample["reviews"])
    return (
        f"You are summarizing customer reviews for product {sample['ProductId']}.\n"
        f"Here are {sample['n_reviews']} individual customer reviews:\n\n{lines}\n\n"
        "Write one overall review summarizing the general customer experience "
        "with this product. Base your summary strictly on the reviews above."
    )

for key, s in samples.items():
    print(f"\n{'='*70}\nSAMPLE: {key}\n{'='*70}")
    print(f"ProductId: {s['ProductId']} | n_reviews: {s['n_reviews']} | avg_score: {s['avg_score']} | score_std: {s['score_std']}")
    prompt = build_prompt(s)
    with open(f"data/prompt_{key}.txt", "w") as f:
        f.write(prompt)
    print(f"Prompt saved to data/prompt_{key}.txt ({len(prompt)} chars)")
    print("\n--- Prompt preview (first 500 chars) ---")
    print(prompt[:500] + "...")

with open("data/synthesis_samples.json", "w") as f:
    json.dump(samples, f, indent=2)
print("\n\nAll sample data saved to data/synthesis_samples.json")
print("Prompts ready. Next: feed these to a real LLM (needs your API key).")
