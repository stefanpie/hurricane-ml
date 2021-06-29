import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pyproj import Geod
# from geopy.distance import geodesic

# sys.path.append(os.path.abspath('../utils'))
# from geo_calculations import vincenty_inverse, haversine

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns


pd.set_option('display.max_columns', 500)


df = pd.read_csv('./data/hurdat/hurdat2_processed.csv')
df = df[df['year'] >= 0]
df = df[(df['system_status'] == 'HU') | (df['system_status'] == 'TS')]
# df = df[df['wind_radii_34_NE'] != -999]
print(list(df.columns))
print(df)

input("wait...")

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
#     for direction in ['NE', 'SE', 'SW', 'NW']:
#         for t in ["-6","-12","-18","-24"]:
#             wind_radii_column_name = 'wind_radii_' + wind_speed + '_' + direction + t
#             input_features.append(wind_radii_column_name)

results_df = pd.DataFrame()

future_steps = [['latitude+12', 'longitude+12'],
                ['latitude+24', 'longitude+24'],
                ['latitude+48', 'longitude+48'],
                ['latitude+72', 'longitude+72'],
                ['latitude+96', 'longitude+96'],
                ['latitude+120', 'longitude+120']]
step_times = ['12','24','36','48','72','96','120']

for future_step, step_time in zip(future_steps, step_times):
    output_features = future_step
    print(future_step)

    x = df[input_features].to_numpy()
    y = df[output_features].to_numpy()
    storm_ids = df['atcf_code'].tolist()

    x_train, x_test, y_train, y_test, storm_ids_train, storm_ids_test = train_test_split(x, y, storm_ids, test_size=0.3, random_state=0)

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)


    reg_linear = LinearRegression()
    reg_svm = SVR(kernel="rbf")
    reg_decision_tree = DecisionTreeRegressor(random_state=0)
    reg_random_forest = RandomForestRegressor(random_state=0, n_jobs=-1)
    reg_gb = GradientBoostingRegressor(random_state=0)

    models = [["Linear", reg_linear],
            ["SVM", reg_svm],
            ["Decision Tree", reg_decision_tree],
            ["Random Forest", reg_random_forest],
            ["Gradient Boosting", reg_gb]]

    for i in range(len(models)):
        models[i][1] = MultiOutputRegressor(models[i][1])

    for m in models:
        print(f"Fitting Model: {m[0]}")
        m[1].fit(x_train, y_train)

    wgs84_geod = Geod(ellps='WGS84')
    def delta_distance_azimuth(lat1,lon1,lat2,lon2):
        az12, az21, dist = wgs84_geod.inv(lon1,lat1,lon2,lat2)
        dist = [x / 1000.0 for x in dist]
        return dist, az12

    for m in models:
        print(f"Model Name: {m[0]}")
        y_pred = m[1].predict(x_test)
        # print(y_test[0])
        # print(y_pred[0])
        distance_error_km, _ = delta_distance_azimuth(y_test[:,0], y_test[:,1], y_pred[:,0], y_pred[:,1])
        distance_error_km = np.array(distance_error_km)
        distance_error_nm = distance_error_km * 0.539957
        print(f"Average of Error Ditance (nm): {distance_error_nm.mean()}")
        print(f"Stdev. Error Ditance (nm): {distance_error_nm.std()}")
        
        results_df['error_distances'+m[0]+'_'+step_time] = distance_error_nm
        print()
        # r2 = metrics.r2_score(y_test, y_pred)
        # mse = metrics.mean_squared_error(y_test, y_pred)
        # rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        # wind_mae = mean_absolute_error(y_test[:,2], y_pred[:,2])
        # pressure_mae = mean_absolute_error(y_test[:,3], y_pred[:,3])
        # print(f"R^2: {r2}")
        # print(f"MSE: {mse}")
        # print(f"RMSE: {rmse}")
        # print(f"Wind MAE: {wind_mae}")
        # print(f"Pressure MAE: {pressure_mae}")

results_df.to_csv('model_errors.csv',index=False)