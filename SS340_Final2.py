#!/usr/bin/env python3


import time
from geopy.geocoders import Nominatim
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.distance import distance, lonlat
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('processed_brain_more.csv')

print(df['Race'].unique())

df['Black'] = 0
df['Asian'] = 0
df.loc[df['Race'] == 'Black or African American', 'Black'] = 1
df.loc[df['Race'] == 'Asian or Pacific Islander', 'Asian'] = 1


print(df.describe())
X = df[['min_dist','Sex', 'Black', 'Asian']]
X = sm.add_constant(X)
est = sm.OLS(df['Crude Rate'], X.astype(float))
est = est.fit(cov_type='HC1')
print(est.summary())