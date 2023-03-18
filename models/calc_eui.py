import pandas as pd


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

# Select columns 4, 5, 6, and 7 and store them in separate lists
electricity = df.iloc[:, 3].tolist()
gas = df.iloc[:, 4].tolist()
oil = df.iloc[:, 5].tolist()
gsf = df.iloc[:, 6].tolist()
eui_list = []

for i in range(0, len(electricity)):
    E = electricity[i]
    G = gas[i]
    O = oil[i]
    GSF = gsf[i]
    eui_list.append(eui(E, G, O, GSF))

print(eui_list)
