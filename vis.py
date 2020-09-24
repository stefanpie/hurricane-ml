import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pyproj import Geod

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns


pd.set_option('display.max_columns', 500)


df = pd.read_csv('./data/hurdat/hurdat2_processed.csv')
# df = df[df['year'] >=0]
# df = df[(df['system_status'] == 'HU') | (df['system_status'] == 'TS')]
# df = df[df['wind_radii_34_NE'] != -999]


input_features = ['year', 'month', 'day', 'hour',

				  'longitude', 'latitude',
				  'x','y',
				  'max_sus_wind', 'min_pressure',
				  'delta_distance', 'azimuth',

				  'longitude-6', 'latitude-6',
				  'x-6','y-6',
				  'max_sus_wind-6', 'min_pressure-6',
				  'delta_distance-6', 'azimuth-6',

				  'longitude-12', 'latitude-12',
				  'x-12','y-12',
				  'max_sus_wind-12', 'min_pressure-12',
				  'delta_distance-12', 'azimuth-12',

                  'longitude-18', 'latitude-18',
				  'x-18','y-18',
				  'max_sus_wind-18', 'min_pressure-18',
				  'delta_distance-18', 'azimuth-18',

                  'longitude-24', 'latitude-24',
				  'x-24','y-24',
				  'max_sus_wind-24', 'min_pressure-24',
				  'delta_distance-24', 'azimuth-24',

				  'day_of_year','jday',
				  'vpre','vpre-6','vpre-12','vpre-18','vpre-24',
				  'landfall','landfall-6','landfall-12','landfall-18','landfall-24']

# for wind_speed in ['34', '50', '64']:
# 	for direction in ['NE', 'SE', 'SW', 'NW']:
# 		for t in ["-6","-12","-18","-24"]:
# 			wind_radii_column_name = 'wind_radii_' + wind_speed + '_' + direction + t
# 			input_features.append(wind_radii_column_name)

print(len(df))

x = df[input_features].to_numpy()
storm_ids = df['atcf_code'].tolist()
years = (df['year'].astype(str) + '_').tolist()
landfall = df['landfall'].tolist()


pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
x_reduced = pca.fit_transform(x)

sns.scatterplot(x=x_reduced[:,0], y=x_reduced[:,1], hue=landfall)
plt.show()