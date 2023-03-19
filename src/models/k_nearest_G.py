import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from src.models.calc_eui import therms_to_kBTu

DATA_PATH = Path("../../data")

features = ["Building Gross Footage", "year_built", "Stories"]
df_G = pd.read_csv(DATA_PATH.joinpath("processed", "df_G_train.csv"))
X = df_G[features].to_numpy()
y = df_G["G(therms)"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for k in range(1, 11):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    MAE = np.abs(y_pred - y_test).mean()
    print(f"Performance for {k} nearest neighbors:")
    print(f"MAE: {MAE}")
