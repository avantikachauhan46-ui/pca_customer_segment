import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "WebsiteVisits",
]

def load_and_prepare_data(path="data/customers.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled
