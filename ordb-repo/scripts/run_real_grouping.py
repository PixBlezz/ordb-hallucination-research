import pandas as pd

print("Loading real Amazon Fine Food Reviews dataset...")
df = pd.read_csv("data/Reviews.csv")
print(f"Loaded {len(df):,} real reviews across {df['ProductId'].nunique():,} unique products.\n")

# Group by product, same logic as pipeline.py's group_reviews()
grouped = df.groupby("ProductId").agg(
    n_reviews=("Score", "count"),
    avg_score=("Score", "mean"),
    score_std=("Score", "std"),
).reset_index()

print("--- Distribution of group sizes (real data) ---")
print(grouped["n_reviews"].describe())

# Filter to groups with enough reviews for our 5/10/20 volume-variation test
for threshold in [5, 10, 20]:
    n_qualifying = (grouped["n_reviews"] >= threshold).sum()
    print(f"\nProducts with >= {threshold} reviews: {n_qualifying:,}")

print("\n--- Top 5 products by review count (real data, highest synthesis volume) ---")
top5 = grouped.sort_values("n_reviews", ascending=False).head(5)
print(top5.to_string(index=False))

print("\n--- Products with highest review-score variance (highest content ambiguity) ---")
high_var = grouped[grouped["n_reviews"] >= 10].sort_values("score_std", ascending=False).head(5)
print(high_var.to_string(index=False))

grouped.to_csv("data/product_groups_real.csv", index=False)
print("\nSaved grouped summary to data/product_groups_real.csv")
