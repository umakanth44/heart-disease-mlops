import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","num"]

df = pd.read_csv(url, names=columns)
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df.drop("num", axis=1, inplace=True)

df.to_csv("data/heart.csv", index=False)
print("Dataset saved to data/heart.csv")
