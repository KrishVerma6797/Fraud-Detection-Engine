import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    X = df.drop(["fraud", "transaction_id"], axis=1)
    y = df["fraud"]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler