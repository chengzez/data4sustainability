import pandas as pd
from pathlib import Path

DATA_PATH = Path('../../data')

df_residential = pd.read_csv(DATA_PATH.joinpath('raw', 'residential.csv'), header=1)
df_research = pd.read_csv(DATA_PATH.joinpath('raw', 'research.csv'), header=1)

df_concat = pd.concat([df_residential, df_research], axis=0)
df_concat.to_csv(DATA_PATH.joinpath('raw', 'concat_data.csv'), index=False)


