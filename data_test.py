import pandas as pd
import numpy as np

data_toliara = pd.read_csv('data/toliara_hourly_completed.csv', sep=",")
print(data_toliara["PS"].mean)