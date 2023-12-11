import csv
import numpy as np
import pandas as pd

data_src = r'../data/TEPDataset/d00.dat'

data_table = pd.read_csv(data_src, header=None, encoding='utf-8', delimiter=r'\s+', quoting=csv.QUOTE_NONE)

data = data_table.to_numpy()

print(data_table)

