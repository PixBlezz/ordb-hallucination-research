import pandas as pd

df = pd.read_csv("data/Reviews.csv")
grouped = df.groupby("ProductId").agg(n=("Score","count"), sd=("Score","std"), avg=("Score","mean")).reset_index()

batch = []
c1 = grouped[(grouped["n"]>=5)&(grouped["n"]<=6)&(grouped["sd"]<0.5)].sample(2, random_state=1)
c2 = grouped[(grouped["n"]>=5)&(grouped["n"]<=6)&(grouped["sd"]>1.5)].sample(2, random_state=1)
c3 = grouped[(grouped["n"]>=10)&(grouped["n"]<=12)&(grouped["sd"]<0.5)].sample(2, random_state=1)
c4 = grouped[(grouped["n"]>=10)&(grouped["n"]<=12)&(grouped["sd"]>1.5)].sample(2, random_state=1)
c5 = grouped[(grouped["n"]>=19)&(grouped["n"]<=22)&(grouped["sd"]<0.5)].sample(2, random_state=1)
c6 = grouped[(grouped["n"]>=19)&(grouped["n"]<=22)&(grouped["sd"]>1.3)].sample(2, random_state=1)

batch_df = pd.concat([c1,c2,c3,c4,c5,c6]).reset_index(drop=True)
print(batch_df.to_string(index=False))
batch_df.to_csv("data/batch_selection.csv", index=False)
