import pandas as pd
import numpy as np
from pathlib import Path
from src.data.get_distances import find_dist_to_GSU


DATA_PATH = Path("../../data")

df = pd.read_csv(DATA_PATH.joinpath("predictions.csv"))

df["Address"] = df["Address"].apply(lambda a: str(a).lower() + ", boston, ma")

df["distance_from_GSU (mi)"] = df["Address"].apply(
    lambda loc1: find_dist_to_GSU(loc1)
)

df.to_csv(DATA_PATH.joinpath("predictions_processed.csv"), index=False)