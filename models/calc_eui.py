import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def kWh_to_kBTu(electricity):
    "Converts kilowatt-hours(kWh) to kilobtus(kBtu)"
    return electricity*3.4121


def therms_to_kBTu(gas):
    "Converts therms(thm) to kilobtus(kBtu)"
    return gas*99.976


def gallons_to_kBtu(oil):
    "Converts gallons to kilobtus(kBtu)"
    return oil*139


def eui(E, G, O, GSF):
    """
    Calculates energy use intensity given energy uses and building gross square footage. 
    : param E - electricity use in kBtu
    : param G - gas use in kBtu
    : param O - natural oil use in kBtu
    : param GSF - building gross square footage
    """
    annual_energy_use = E + G + O
    return annual_energy_use/GSF


# Read the CSV file into a pandas dataframe, skipping the first row (header)
df = pd.read_csv('residential.csv', skiprows=[0])
# df = pd.read_csv('research.csv', skiprows=[0])

electricity = []
gas = []
oil = []
gsf = []
totalEnergy = []
eui_list = []

for index, row in df.iterrows():
    if pd.notna(row[1]):
        electricity.append(row[3])
        gas.append(row[4])
        oil.append(row[5])
        gsf.append(row[6])
# Select columns 4, 5, 6, and 7 and store them in separate lists
# electricity = df.loc[df.iloc[:, 1].notnull(), 3].tolist()
# gas = df.iloc[df.iloc[:, 1].notnull(), 4].tolist()
# oil = df.iloc[df.iloc[:, 1].notnull(), 5].tolist()
# gsf = df.iloc[df.iloc[:, 1].notnull(), 6].tolist()

for i in range(0, len(electricity)):

    E = kWh_to_kBTu(electricity[i])
    G = therms_to_kBTu(gas[i])
    O = gallons_to_kBtu(oil[i])
    GSF = gsf[i]
    if ():
        print("yes")
    totalEnergy.append(E + G + O)
    eui_list.append(eui(E, G, O, GSF))

# print(eui_list)
# x = pd.DataFrame(gsf).reshape(-1, 1)
# y = pd.DataFrame(totalEnergy).reshape(-1, 1)
x = np.array(gsf).reshape(-1, 1)
y = np.array(totalEnergy).reshape(-1, 1)
reg = LinearRegression().fit(x, y)
y_pred = reg.predict(x)
plt.scatter(gsf, totalEnergy)
plt.plot(x, y_pred, color='red')
plt.xlabel('GSF')
plt.ylabel('Total Energy')
plt.title('Scatter Plot of GSF vs Total Energy')
plt.show()
