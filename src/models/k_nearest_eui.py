import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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

DATA_PATH = Path("../../data")

features = ["year_built", "Stories"]
df = pd.read_csv(DATA_PATH.joinpath("processed", "df_train.csv"))
X = df[features].to_numpy()
y = df["eui"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

metrics = {'k': [], 'MAE': []}
for k in range(1, 11):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)
    # X_test = scaler.transform(X_test)
    y_pred = neigh.predict(X_test)
    MAE = np.abs(y_pred - y_test).mean()
    r2 = neigh.score(X_test, y_test)
    metrics['k'].append(k)
    metrics['MAE'].append(MAE)
    print(f"R^2: {r2}")
    print(f"Performance for {k} nearest neighbors:")
    print(f"MAE: {MAE}")
    
fig = plt.figure(figsize=(7.5, 10))
ax = fig.add_subplot(111)
ax.plot(metrics['k'], metrics['MAE'])
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('K-Nearest Neighbors with Different K Values')
# plt.show()
fig.savefig('../../reports/figures/knn.svg', bbox_inches='tight')
    

    

