import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from pyproj import Geod

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns


pd.set_option('display.max_columns', 500)


df = pd.read_csv('./data/hurdat/hurdat2_processed.csv')
df = df[df['year'] >= 2000]
df = df[(df['system_status'] == 'HU') | (df['system_status'] == 'TS')]
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
				  'landfall','landfall-6','landfall-12','landfall-18','landfall-24',]

# for wind_speed in ['34', '50', '64']:
# 	for direction in ['NE', 'SE', 'SW', 'NW']:
# 		for t in ["-6","-12","-18","-24"]:
# 			wind_radii_column_name = 'wind_radii_' + wind_speed + '_' + direction + t
# 			input_features.append(wind_radii_column_name)

output_features = ['latitude+48', 'longitude+48']


x = df[input_features].to_numpy()
y = df[output_features].to_numpy()
storm_ids = df['atcf_code'].tolist()



x_train, x_test, y_train, y_test, storm_ids_train, storm_ids_test = train_test_split(x, y, storm_ids, test_size=0.3, random_state=0)

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)


model = keras.Sequential(
    [
        layers.Dense(1024, activation=activations.elu, input_shape=(170,)),
		layers.Dense(1024, activation=activations.elu),
		layers.Dense(1024, activation=activations.elu),
		layers.Dense(1024, activation=activations.elu),
		layers.BatchNormalization(),
        layers.Dense(512, activation=activations.elu),
        layers.Dense(512, activation=activations.elu),
		layers.BatchNormalization(),
        layers.Dense(256, activation=activations.elu),
        layers.Dense(256, activation=activations.elu),
		layers.BatchNormalization(),
        layers.Dense(128, activation=activations.elu),
        layers.Dense(128, activation=activations.elu),
		layers.BatchNormalization(),
        layers.Dense(2, activation="linear")
    ]
)

def haversine(y_true, y_pred):
	pi_on_180 = 0.017453292519943295

	lat1 = y_true[0]*pi_on_180
	lat2 = y_pred[0]*pi_on_180

	lon1 = y_true[1]*pi_on_180
	lon2 = y_pred[1]*pi_on_180

	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = tf.math.sin(dlat/2)**2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon/2)**2
	c = 2 * tf.math.asin(tf.math.sqrt(a)) 

	km = 6371* c
	return km

def equirectangular_distance(y_true, y_pred):
	pi_on_180 = 0.017453292519943295

	lat1 = y_true[0]*pi_on_180
	lat2 = y_pred[0]*pi_on_180

	lon1 = y_true[1]*pi_on_180
	lon2 = y_pred[1]*pi_on_180

	x = (lon2 - lon1) * tf.math.cos( 0.5*(lat2 + lat1) )
	y = lat2 - lat1
	d = 6371 * tf.math.sqrt( x*x + y*y )
	return d


model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=[equirectangular_distance,haversine,"mse", "mae"])

batch_size = 64
epochs = 200
csv_logger = callbacks.CSVLogger('training.csv')

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batch_size,epochs=epochs, callbacks=[csv_logger])
y_pred = model.predict(x_test)

wgs84_geod = Geod(ellps='WGS84')
def delta_distance_azimuth(lat1,lon1,lat2,lon2):
    az12, az21, dist = wgs84_geod.inv(lon1,lat1,lon2,lat2)
    dist = [x / 1000.0 for x in dist]
    return dist, az12

distance_error_km, _ = delta_distance_azimuth(y_test[:,0], y_test[:,1], y_pred[:,0], y_pred[:,1])
distance_error_km = np.array(distance_error_km)
distance_error_nm = distance_error_km * 0.539957
print(distance_error_nm.mean())
print(distance_error_nm.std())

# wind_mae = mean_absolute_error(y_test[:,2], y_pred[:,2])
# print(f"Wind MAE: {wind_mae}")


plt.hist(distance_error_nm, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Error Distance (nm)')
plt.title(r'Histogram of Error Distance (nm)')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()