import pandas as pd
import numpy as np
from datetime import timedelta
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def data_collect():

    confirmed_df_temp = pd.read_csv('data/time_series_covid19_confirmed_global.csv')
    deaths_df_temp = pd.read_csv('data/time_series_covid19_deaths_global.csv')
    recovered_df_temp = pd.read_csv('data/time_series_covid19_recovered_global.csv')

    def preproc_frame(df):
        df.replace(np.nan, '', regex=True, inplace=True)
        df['Country'] = df['Country/Region'] + ' ' + df['Province/State']
        df.drop(columns=['Country/Region', 'Province/State'], inplace=True)
        df.set_index('Country', inplace=True)
        return df

    confirmed_df = preproc_frame(confirmed_df_temp)
    deaths_df = preproc_frame(deaths_df_temp)
    recovered_df = preproc_frame(recovered_df_temp)

    return confirmed_df, deaths_df, recovered_df



def task_1(confirmed_df, deaths_df, recovered_df):

    df_coordinates = confirmed_df.loc[:, ['Lat', 'Long']]
    df_confirmed = confirmed_df.drop(columns=['Lat', 'Long'])
    df_confirmed.columns = pd.to_datetime(df_confirmed.columns)
    df_deaths = deaths_df.drop(columns=['Lat', 'Long'])
    df_deaths.columns = pd.to_datetime(df_deaths.columns)
    df_recovered = recovered_df.drop(columns=['Lat', 'Long'])
    df_recovered.columns = pd.to_datetime(df_recovered.columns)


    diff1 = list(set(df_confirmed.index.values) - set(df_deaths.index.values))
    # print(diff1)
    diff2 = list(set(df_confirmed.index.values) - set(df_recovered.index.values))
    for name in diff2:
        df_recovered.loc[name, :] = 0
        for date in df_recovered.columns[14:]:
            active = df_confirmed.loc[name, date] - df_confirmed.loc[name, (date - timedelta(days=14))]
            dead = df_deaths.loc[name, date] - df_deaths.loc[name, (date - timedelta(days=14))]
            active = active - dead
            res = df_confirmed.loc[name, date] - df_deaths.loc[name, date] - active
            df_recovered.loc[name, date] = res


    df_deaths['sum'] = df_deaths.sum(axis=1)
    list_to_drop = df_deaths[df_deaths['sum'] == 0].index
    df_deaths = df_deaths.drop(index=list_to_drop)
    df_recovered = df_recovered.drop(index=list_to_drop)
    df_confirmed = df_confirmed.drop(index=list_to_drop)
    df_coordinates = df_coordinates.drop(index=list_to_drop)
    df_deaths = df_deaths.drop(columns=['sum'])


    df_deaths.columns = pd.to_datetime(df_deaths.columns)
    df_deaths_monthly = df_deaths.groupby([df_deaths.columns.year, df_deaths.columns.month], axis=1).sum()
    df_recovered.columns = pd.to_datetime(df_recovered.columns)
    df_recovered_monthly = df_recovered.groupby([df_recovered.columns.year, df_recovered.columns.month], axis=1).sum()
    df_recovered_copy = df_recovered_monthly.copy()
    df_recovered_copy[df_recovered_monthly == 0] = 1
    mortality = df_deaths_monthly / df_recovered_copy
    mortality[mortality > 1.0] = 1.0
    print('\nZADANIE 1 - śmiertelność: ')
    print(mortality)

    active_cases = df_confirmed - df_deaths - df_recovered
    print('\nZADANIE 1 - liczba aktywnych przypadków: ')
    print(active_cases)
    #odrzucenie danych dla mniej niz 100 aktywnych przypadkow zostalo zaimplementowane w task_2
    return active_cases, df_coordinates



def task_2(active_cases):
    temp = active_cases.copy()
    active_ov100 = temp[temp >= 100].fillna(0)
    counting = active_ov100.copy()
    counting[active_ov100 > 0] = 1

    active_cases_copy = active_ov100.copy()
    counting_copy = counting.copy()

    for i in range(7, len(active_ov100.columns.values)):
        for j in range(0, 6):
            active_cases_copy.loc[:, active_cases_copy.columns[i]] += active_ov100.loc[:, active_cases_copy.columns[i - j]]
            counting_copy.loc[:, active_cases_copy.columns[i]] += counting.loc[:, active_cases_copy.columns[i - j]]

    cnt = counting_copy.copy()
    cnt[counting_copy == 0] = 1
    M = active_cases_copy / cnt

    M_div = M.copy()
    M_div[M == 0] = 1
    R = M.copy()
    for i in range(5, len(M.columns.values)):
        R.loc[:, active_cases_copy.columns[i]] = M.loc[:, active_cases_copy.columns[i]] / M_div.loc[:, active_cases_copy.columns[i - 5]]

    return R



def weather(R, df_coordinates):
    weather_max = Dataset('./data/TerraClimate_tmax_2018.nc')
    weather_min = Dataset('./data/TerraClimate_tmin_2018.nc')

    coor_idx_lat_max = pd.DataFrame(weather_max['tmax'][0]).shape[0]
    coor_idx_lon_max = pd.DataFrame(weather_max['tmax'][0]).shape[1]

    rescaled_cols = []
    for col in range(coor_idx_lon_max):
        rescaled_cols.append(int(((180.0 + 180.0) * col / coor_idx_lon_max) - 180))
    rescaled_rows = []
    for row in range(coor_idx_lat_max):
        rescaled_rows.append(int(((90.0 + 90) * row / coor_idx_lat_max) - 90) * (-1))

    R.columns = pd.to_datetime(R.columns)
    R_20 = R.groupby([R.columns.year, R.columns.month], axis=1).sum()[2020]
    temp_mean = R_20.copy()
    R_20 = pd.concat([R_20, df_coordinates], axis=1)
    R_20.dropna(inplace=True)

    for month in range(0, 12):
        df_w_max = pd.DataFrame(weather_max['tmax'][month])
        df_w_min = pd.DataFrame(weather_min['tmin'][month])
        temp_mean_monthly = (df_w_max + df_w_min) / 2
        temp_mean_monthly.index = rescaled_rows
        temp_mean_monthly.columns = rescaled_cols
        for index in range(len(R_20.index.values)-1):
            temp_mean.iloc[index, month] = temp_mean_monthly.loc[int(R_20.iloc[index, -2]), int(R_20.iloc[index, -1])].iloc[0, 0]

    return R_20, temp_mean



def hypothesis_task1(R, temp_mean):
    R.drop(columns=['Lat', 'Long'], inplace=True)
    print(R)
    print(temp_mean)



if __name__ == '__main__':
    confirmed_df, deaths_df, recovered_df = data_collect()
    active_cases, df_coordinates = task_1(confirmed_df, deaths_df, recovered_df)
    R = task_2(active_cases)
    R_20, temp_mean = weather(R, df_coordinates)
    hypothesis_task1(R_20, temp_mean)