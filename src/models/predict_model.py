import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

DATA_PATH = Path("../../data")

features = ["year_built", "Stories"]
df = pd.read_csv(DATA_PATH.joinpath("processed", "df_train.csv"))
X = df[features].to_numpy()
y = df["eui"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

neigh = KNeighborsRegressor(n_neighbors=6)
neigh.fit(X_train, y_train)

df_test = pd.read_csv(DATA_PATH.joinpath("predictions_processed.csv"))
X_test = df_test[features].to_numpy()
df_test["Predicted_EUI"] = neigh.predict(X_test)
df_test.to_csv(DATA_PATH.joinpath("final_predictions.csv"))

