"""
ORDB (Overall Review Detection Benchmark) pipeline.

Steps:
1. Load review data (mock data here; swap load_mock_data() for a real
   pd.read_csv("Reviews.csv") once you have the Amazon Fine Food Reviews
   dataset downloaded from Kaggle).
2. Group reviews by ProductId, varying group size (5/10/20) to test
   volume-dependence of hallucination risk.
3. Build the synthesis prompt for each group (the step that hallucinated
   in the original donation-platform project).
4. Call an LLM (Gemini, or another model) to generate the
   synthesized "overall review" for each group -- requires an API key,
   not available in this sandboxed environment.
5. Scaffold for manual labeling using the MiRANews intrinsic/extrinsic
   hallucination taxonomy.
"""

import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# STEP 1: Mock data (structured exactly like Amazon Fine Food Reviews'
# real columns: ProductId, Score, Summary, Text)
# ---------------------------------------------------------------------
def load_mock_data() -> pd.DataFrame:
    rows = [
        # Product A: consistent, mildly positive reviews -- low hallucination risk expected
        {"ProductId": "A001", "Score": 4, "Summary": "Good but pricey",
         "Text": "Tastes great, a bit expensive for the size."},
        {"ProductId": "A001", "Score": 4, "Summary": "Solid choice",
         "Text": "Good flavor, arrived fresh, would buy again."},
        {"ProductId": "A001", "Score": 3, "Summary": "Okay",
         "Text": "It's fine, nothing special, a bit too sweet for me."},
        {"ProductId": "A001", "Score": 5, "Summary": "Excellent",
         "Text": "Best in this category I've tried, will reorder."},
        {"ProductId": "A001", "Score": 4, "Summary": "Would recommend",
         "Text": "Good quality, packaging was solid too."},

        # Product B: mixed/contradictory reviews -- higher hallucination risk expected
        {"ProductId": "B002", "Score": 1, "Summary": "Arrived stale",
         "Text": "Package was open, product was stale, disappointed."},
        {"ProductId": "B002", "Score": 5, "Summary": "Perfect",
         "Text": "Fresh, well packaged, exactly as described."},
        {"ProductId": "B002", "Score": 2, "Summary": "Not as described",
         "Text": "Smaller portion than the listing implied."},
        {"ProductId": "B002", "Score": 4, "Summary": "Pretty good",
         "Text": "Tasted fine, no issues with my order."},
        {"ProductId": "B002", "Score": 1, "Summary": "Broken seal",
         "Text": "Seal was broken on arrival, requested a refund."},
        {"ProductId": "B002", "Score": 5, "Summary": "Great value",
         "Text": "Cheaper than the store and just as good."},
        {"ProductId": "B002", "Score": 3, "Summary": "Average",
         "Text": "Nothing wrong with it, nothing exciting either."},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# STEP 2: Group by product, keep group size as a variable (this IS the
# "volume" covariate from the hazard-model discussion)
# ---------------------------------------------------------------------
def group_reviews(df: pd.DataFrame) -> dict:
    groups = {}
    for product_id, group_df in df.groupby("ProductId"):
        groups[product_id] = {
            "n_reviews": len(group_df),
            "avg_score": round(group_df["Score"].mean(), 2),
            "score_std": round(group_df["Score"].std(), 2) if len(group_df) > 1 else 0.0,
            "reviews": group_df[["Score", "Summary", "Text"]].to_dict("records"),
        }
    return groups


# ---------------------------------------------------------------------
# STEP 3: Build the synthesis prompt -- this is the exact prompt
# structure whose OUTPUT is what you'd check for hallucination
# ---------------------------------------------------------------------
def build_synthesis_prompt(product_id: str, group: dict) -> str:
    review_lines = "\n".join(
        f"- (Rating {r['Score']}/5) {r['Summary']}: {r['Text']}"
        for r in group["reviews"]
    )
    prompt = (
        f"You are summarizing customer reviews for product {product_id}.\n"
        f"Here are {group['n_reviews']} individual customer reviews:\n\n"
        f"{review_lines}\n\n"
        "Write one overall review summarizing the general customer experience "
        "with this product. Base your summary strictly on the reviews above."
    )
    return prompt


# ---------------------------------------------------------------------
# STEP 4: PLACEHOLDER for the actual LLM call.
# Requires a real API key (Gemini, or Claude via the Anthropic API) --
# not available in this sandboxed environment. Fill this in when running
# locally / with your own API access.
# ---------------------------------------------------------------------
def generate_overall_review(prompt: str) -> str:
    raise NotImplementedError(
        "Plug in your LLM API call here (e.g., Gemini API, matching the "
        "original donation-platform setup). Example:\n\n"
        "  import google.generativeai as genai\n"
        "  genai.configure(api_key=YOUR_KEY)\n"
        "  model = genai.GenerativeModel('gemini-pro')\n"
        "  return model.generate_content(prompt).text\n"
    )


# ---------------------------------------------------------------------
# STEP 5: Labeling scaffold (MiRANews intrinsic/extrinsic taxonomy)
# You fill this in BY HAND after reading each synthesized review against
# its source reviews -- this is not automatable without another model
# doing the judging, which introduces its own reliability question.
# ---------------------------------------------------------------------
@dataclass
class HallucinationLabel:
    product_id: str
    n_reviews_synthesized: int
    hallucinated: bool = False
    hallucination_type: str = None  # "intrinsic", "extrinsic", or None
    unsupported_claim: str = ""
    notes: str = ""


def print_pipeline_summary(groups: dict):
    print(f"{'ProductId':<10} {'#Reviews':<10} {'AvgScore':<10} {'ScoreStd':<10}")
    print("-" * 42)
    for pid, g in groups.items():
        print(f"{pid:<10} {g['n_reviews']:<10} {g['avg_score']:<10} {g['score_std']:<10}")


if __name__ == "__main__":
    df = load_mock_data()
    print(f"Loaded {len(df)} mock reviews across {df['ProductId'].nunique()} products.\n")

    groups = group_reviews(df)
    print_pipeline_summary(groups)

    print("\n--- Example synthesis prompt (Product B002, higher variance) ---\n")
    prompt = build_synthesis_prompt("B002", groups["B002"])
    print(prompt)

    print("\n--- Next step (requires API key, not run here) ---")
    print("generate_overall_review(prompt) would call Gemini/Claude here.")
