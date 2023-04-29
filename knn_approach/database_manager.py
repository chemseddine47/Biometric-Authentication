import pandas as pd


def add_to_dataset(row, row2):
    df = pd.read_csv('dataset.csv')
    df = df.append(row, ignore_index=True)
    df.to_csv('dataset.csv', index=False)

    df2 = pd.read_csv('dataset2.csv')
    df2 = df2.append(row2, ignore_index=True)
    df2.to_csv('dataset2.csv', index=False)


def extract_from_dataset():
    df = pd.read_csv('dataset.csv')
    df2 = pd.read_csv('dataset2.csv')
    return df, df2
