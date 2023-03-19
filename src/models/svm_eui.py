import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from src.models.calc_eui import eui
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
import matplotlib_inline
# get higher quality plots
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
import shap

DATA_PATH = Path("../../data")

features = ["year_built", "Stories"]
df_E = pd.read_csv(DATA_PATH.joinpath("processed", "df_train.csv"), index_col=0)
# features = df_E.drop(["eui"], axis=1).columns
print(features)
X = df_E[features].to_numpy()
# X = df_E[features].to_numpy()
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
# r2 = reg.score(X_test, y_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2: {r2}")
print(f"Performance for SVM:")
print(f"MAE: {MAE}")

# ex = shap.KernelExplainer(reg.predict, X_train)
# shap_values = ex.shap_values(X_train)
# fig = shap.summary_plot(shap_values, X_train, feature_names=features, show=False)
# plt.savefig('../../reports/figures/shap_svr.svg', bbox_inches='tight')
