import os
import pandas as pd

def load_and_describe_data():
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cardio_train.csv')
    df = pd.read_csv(file_path, sep=';')
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nNull Counts:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    return df

if __name__ == '__main__':
    load_and_describe_data()
