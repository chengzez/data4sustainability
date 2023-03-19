import pandas as pd
import numpy as np
from pathlib import Path
from thefuzz import process
import re
from src.data.get_distances import find_dist_to_GSU


DATA_PATH = Path("../../data")

df = pd.read_csv(DATA_PATH.joinpath("raw", "concat_data.csv"))

df["Address"] = df["Address"].apply(lambda a: str(a).lower() + ", boston, ma")

# KNN features
# Geographic Location (have feature)
# Building Code (have feature)
# Property Type (have feature)

# Year Built (have feature but needs to be processed)
# Square Footage of Building (have feature but needs to be processed)
# Land Area (have feature but needs to be processed)
# Height of Building (ft) (have feature but needs to be processed)
# Stories of Building (B+3 is 4) (have feature but needs to be processed)

# Predict either EUI or [E, U, I]

residential = pd.read_excel(
    DATA_PATH.joinpath("raw", "Table4-1 List of Residential Properties.xlsx"), header=1
)
research = pd.read_excel(
    DATA_PATH.joinpath(
        "raw", "Table4-2 List of Academic, Administrative and Other Properties.xlsx"
    ),
    header=2,
)

residential = residential.drop(labels=["Building SF"], axis=1).rename(
    {"Property Address": "Address", "Year\nBuilt": "year_built"}, axis=1
)

research = research.drop(labels=["Bldg Area", "Use", "Land\nArea"], axis=1).rename(
    {"Property Address": "Address", "Year\nBuilt": "year_built"}, axis=1
)

residential["Address"] = residential["Address"].apply(
    lambda s: str(s).split()[0] + " Commonwealth Avenue," + " ".join(str(s).split()[3:])
    if str(s).find("Comm") >= 0
    else str(s)
)
research["Address"] = research["Address"].apply(
    lambda s: str(s).split()[0] + " Commonwealth Avenue," + " ".join(str(s).split()[3:])
    if str(s).find("Comm") >= 0
    else str(s)
)

residential["Address"] = residential["Address"].apply(
    lambda s: str(s).replace("Saint", "St.")
)
research["Address"] = research["Address"].apply(
    lambda s: str(s).replace("Saint", "St.")
)


df["Address"] = df["Address"].apply(
    lambda s: str(s).split()[0] + " Commonwealth Avenue," + " ".join(str(s).split()[3:])
    if str(s).find("Comm") >= 0
    else str(s)
)
df["Address"] = df["Address"].apply(
    lambda s: str(s).split()[0] + " Commonwealth Avenue," + " ".join(str(s).split()[3:])
    if str(s).find("Comm") >= 0
    else str(s)
)

df["Address"] = df["Address"].apply(lambda s: str(s).replace("Saint", "St."))

df_concat = pd.concat([residential, research], axis=0)

df_concat["Address"] = df_concat["Address"].str.lower()
filt = df_concat.Address.str[0].apply(lambda a: str(a).isdigit())
df_concat = df_concat.loc[filt]
df_concat["Address"] = (
    df_concat["Address"]
    .apply(lambda a: str(a).split("-")[0])
    .apply(lambda a: str(a).split(",")[0])
)
df_concat = df_concat[df_concat["Address"].apply(len) > 3]
df_concat = df_concat.dropna()
df_concat["Height"] = (
    df_concat["Height"]
    .str.replace("['’]", "", regex=True)
    .str.replace('"', "")
    .apply(lambda s: str(s).split()[0])
)

df_concat["year_built"] = pd.to_numeric(df_concat["year_built"], errors="coerce")
df_concat["Height"] = pd.to_numeric(df_concat["Height"], errors="coerce")
df_concat = df_concat.dropna()

df_concat["Stories"] = df_concat.loc[:, "Stories"].apply(
    lambda x: str(x).count("B")
    + str(x).count("M")
    + str(x).count("P")
    + int(re.findall(r"\d+", str(x))[0])
)

df_concat.loc[:, "Address"] = df_concat.loc[:, "Address"] + ", boston, ma"

df["Address"] = df["Address"].apply(
    lambda s: str(s).split("(")[0].replace(", boston, ma", "") + ", boston, ma"
)
df_concat["Address"] = df_concat["Address"].apply(
    lambda s: str(s).split("(")[0].replace(", boston, ma", "") + ", boston, ma"
)

# Create new column in df1 with best match for each value in df2
df["best_match"] = df.loc[:, "Address"].apply(
    lambda x: process.extractOne(x, df_concat["Address"])[0:2]
)
similarities = [val[1] for val in df["best_match"]]
df["match_val"] = similarities
df["best_match"] = df.loc[:, "best_match"].apply(lambda x: x[0])

# Create new column in df2 with best match for each value in df1
df_concat["best_match"] = df_concat.loc[:, "Address"].apply(
    lambda x: process.extractOne(x, df["Address"])[0:2]
)
similarities = [val[1] for val in df_concat["best_match"]]
df_concat["match_val"] = similarities
df_concat["best_match"] = df_concat.loc[:, "best_match"].apply(lambda x: x[0])

df = df.loc[df.match_val >= 0.75, :]
df_concat = df_concat.loc[df_concat.match_val >= 0.75, :]

df_merged = pd.merge(left=df, right=df_concat, how="inner", on="best_match")
df_merged = df_merged[(df_merged.match_val_x >= 95) & (df_merged.match_val_y >= 99)]
df_merged = df_merged[~df_merged["Building Code"].duplicated()].reset_index(drop=True)

df_merged["distance_from_GSU (mi)"] = df_merged["Address_y"].apply(
    lambda loc1: find_dist_to_GSU(loc1)
)

df_merged[
    [
        "Building Code",
        "Property Type",
        "distance_from_GSU (mi)",
        "year_built",
        "Stories",
        "Height",
        "E (kWh)",
        "G(therms)",
        "O(gallon)",
        "Building Gross Footage",
    ]
].to_csv(DATA_PATH.joinpath("processed", "KNN_data.csv"), index=False)
