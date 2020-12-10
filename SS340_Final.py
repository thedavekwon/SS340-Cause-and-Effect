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

# manual parsing for some MSA
cache = {'Augusta, GA':(-81.966667, 33.466667), 'Birmingham, AL':(-86.779633, 33.543682), 'Denver, CO': (-104.991531, 39.742043)}
geolocator = Nominatim(user_agent="myGeocoder")

def convert_to_lat(loc):
  if loc in cache:
    return cache[loc][1]
  else:
    location = geolocator.geocode(loc)
    time.sleep(0.5)
    cache[loc] = (location.longitude, location.latitude)
    return cache[loc][1]

def convert_to_long(loc):
  if loc in cache:
    return cache[loc][0]
  else:
    location = geolocator.geocode(loc)
    time.sleep(0.5)
    cache[loc] = (location.longitude, location.latitude)
    return cache[loc][0]
  

df = pd.read_csv('preproc.csv', delimiter='\t')
df.drop(['Notes', 'MSA Code'], axis=1, inplace=True)
df = df.dropna()
df = df[df['MSA'] != 'Other']
# Does not consider Haiwai
df = df[~df['MSA'].isin(['Honolulu, HI', 'Urban Honolulu, HI'])]
df = df[df['Crude Rate'] != 'Not Applicable']

df.loc[df['Sex'] == "Female", ['Sex']] = 0 
df.loc[df['Sex'] == "Male", ['Sex']] = 1

print(df['MSA'].unique())


def parse_loc(loc):
  city, state = loc.split(',')
  return city.split('-')[0].strip()+", "+state.split('-')[0].strip()
df['MSA'] = df['MSA'].apply(parse_loc)
df['MSA'].replace({'Winston, NC': 'Winston-Salem, NC'}, inplace=True)


df['latitude'] = df['MSA'].apply(convert_to_lat)
df['longitude'] = df['MSA'].apply(convert_to_long)



pdf = pd.read_csv('global_power_plant_database.csv')
pdf = pdf[pdf['country'] == 'USA']
pdf = pdf[pdf['primary_fuel'] == 'Nuclear']


dist_cache = {}
def get_shortest_distance(cur, plants):
  if cur in dist_cache:
    return dist_cache[cur]
  min_dist = min(map(lambda plant: distance(lonlat(*cur), lonlat(*plant)).miles, plants))
  dist_cache[cur] = min_dist
  return min_dist


power_plants_lonlat = list(zip(pdf.longitude, pdf.latitude))
df['min_dist'] = list(map(lambda x: get_shortest_distance(x, power_plants_lonlat), (zip(df.longitude, df.latitude))))
df['min_dist_squared'] = df['min_dist']**2
df['min_dist'].describe()
df['Crude Rate'] = df['Crude Rate'].astype(float)


df.to_csv('processed_net.csv', index=False) 

X = df[['min_dist','Sex']]
X = sm.add_constant(X)
est = sm.OLS(df['Crude Rate'], X.astype(float))
est = est.fit(cov_type='HC1')
print(est.summary())