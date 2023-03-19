

def kWh_to_kBTu(electricity):
    "Converts kilowatt-hours(kWh) to kilobtus(kBtu)"
    return electricity * 3.4121


# example use case
# df['E'] = df['E'].apply(kWh_to_kBTu)


def therms_to_kBTu(gas):
    "Converts therms(thm) to kilobtus(kBtu)"
    return gas * 99.976


# example use case
# df['G'] = df['G'].apply(therms_to_kBTu)


def gallons_to_kBtu(oil):
    "Converts gallons to kilobtus(kBtu)"
    return oil * 139


# example use case
# df['O'] = df['O'].apply(galloons_to_kBTu)


def eui(E, G, O, GSF):
    """
    Calculates energy use intensity given energy uses and building gross square footage.
    : param E - electricity use in kBtu
    : param G - gas use in kBtu
    : param O - natural oil use in kBtu
    : param GSF - building gross square footage
    """
    annual_energy_use = kWh_to_kBTu(E) + therms_to_kBTu(G) + gallons_to_kBtu(O)
    return annual_energy_use / GSF
