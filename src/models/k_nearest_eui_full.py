import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import MinMaxScaler
from src.models.calc_eui import eui

DATA_PATH = Path("../../data")

features = ["year_built", "Stories"]
df_E = pd.read_csv(DATA_PATH.joinpath("processed", "df_train_full.csv"))
X = df_E[features].to_numpy()
y = df_E["eui"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

for k in range(1, 11):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)
    # X_test = scaler.transform(X_test)
    y_pred = neigh.predict(X_test)
    MAE = np.abs(y_pred - y_test).mean()
    print(neigh.score(X_test, y_test))
    print(f"Performance for {k} nearest neighbors:")
    print(f"MAE: {MAE}")
