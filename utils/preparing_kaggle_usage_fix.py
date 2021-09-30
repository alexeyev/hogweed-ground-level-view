# coding:utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = "../"
df = pd.read_csv(ROOT + "prepared_data/gold.csv")

print(df)

public, private = train_test_split(
    df,
    train_size=0.6,
    random_state=100,
    shuffle=True,
    stratify=df.has_hogweed)

print(public.shape)
print(private.shape)
public_ids = set(public.id)

df["Usage"] = df["id"].map(lambda x: "Public" if x in public_ids else "Private")

df.to_csv(ROOT + "prepared_data/gold_with_usage.csv", index=None)
