import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from src.models.calc_eui import eui

DATA_PATH = Path("../../data")

features = ["year_built", "Stories"]
df_E = pd.read_csv(DATA_PATH.joinpath("processed", "df_train_full.csv"))
X = df_E[features].to_numpy()
y = df_E["eui"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


reg = SVR(kernel="rbf", C=1e3, gamma=0.1)
reg.fit(X_train, y_train)
X_test = scaler.transform(X_test)
y_pred = reg.predict(X_test)
MAE = np.abs(y_pred - y_test).mean()
print(reg.score(X_test, y_test))
print(f"Performance for SVM:")
print(f"MAE: {MAE}")
