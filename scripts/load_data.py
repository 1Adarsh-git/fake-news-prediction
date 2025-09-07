import pandas as pd

def load_data(path="data/News.csv"):
    data = pd.read_csv(path, index_col=0)
    data = data.drop(["title", "subject", "date"], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    return data
