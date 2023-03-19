import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from sklearn.preprocessing import MinMaxScaler
from src.models.calc_eui import eui
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

df = pd.read_csv(DATA_PATH.joinpath("processed", "df_train.csv"), index_col=0)
features = ["year_built"]
# features = df.drop(["eui"], axis=1).columns
# print(features)
X = df[features].to_numpy()
y = df["eui"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

metrics = {'MAE': []}
reg = LinearRegression()
reg.fit(X_train, y_train)
# X_test = scaler.transform(X_test)
y_pred = reg.predict(X_test)
print(y_pred.min(), y_pred.max())
MAE = np.abs(y_pred - y_test).mean()
# r2 = reg.score(X_test, y_test)
r2 = r2_score(y_pred, y_test)
metrics['MAE'].append(MAE)
print(f"R^2: {r2}")
print(f"Performance:")
print(f"MAE: {MAE}")

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
residuals = y_train - reg.predict(X_train)
# ax.scatter(X_train, residuals)
ax.axhline(y=y_train.mean(), color='red')
ax.scatter(X_train, y_train)
ax.set_ylabel('EUI')
ax.set_xlabel('Year Built')
plt.savefig('../../reports/figures/lin_reg.svg', bbox_inches='tight')
# plt.show()

# ex = shap.KernelExplainer(reg.predict, X_train)
# shap_values = ex.shap_values(X_train)
# fig = shap.summary_plot(shap_values, X_train, feature_names=features, show=False)
# plt.savefig('../../reports/figures/shap_lin_reg.svg', bbox_inches='tight')

    

    

