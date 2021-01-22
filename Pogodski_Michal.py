import pandas as pd
import numpy as np
from datetime import timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency



def data_collect():

    confirmed_df_temp = pd.read_csv('time_series_covid19_confirmed_global.csv')
    deaths_df_temp = pd.read_csv('time_series_covid19_deaths_global.csv')
    recovered_df_temp = pd.read_csv('time_series_covid19_recovered_global.csv')

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


    # UZUPELNIENIE INFORMACJI O OZDROWIENCACH (ZALOZENIE: CZAS INFEKCJI WYNOSI 14 DNI)
    diff2 = list(set(df_confirmed.index.values) - set(df_recovered.index.values))
    for name in diff2:
        df_recovered.loc[name, :] = 0
        for date in df_recovered.columns[14:]:
            active = df_confirmed.loc[name, date] - df_confirmed.loc[name, (date - timedelta(days=14))]
            dead = df_deaths.loc[name, date] - df_deaths.loc[name, (date - timedelta(days=14))]
            active = active - dead
            res = df_confirmed.loc[name, date] - df_deaths.loc[name, date] - active
            df_recovered.loc[name, date] = res


    # ODRZUCANIE KRAJOW, KTORE NIE PUBLIKUJA DANYCH O SMIERTELNOSCI
    df_deaths['sum'] = df_deaths.sum(axis=1)
    list_to_drop = df_deaths[df_deaths['sum'] == 0].index
    df_deaths = df_deaths.drop(index=list_to_drop)
    df_recovered = df_recovered.drop(index=list_to_drop)
    df_confirmed = df_confirmed.drop(index=list_to_drop)
    df_coordinates = df_coordinates.drop(index=list_to_drop)
    df_deaths = df_deaths.drop(columns=['sum'])


    # OBLICZANIE MIESIECZNEJ SMIERTELNOSCI DLA KRAJOW
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


    # OBLICZANIE LICZBY AKTYWNYCH PRZYPADKOW DLA POSZCZEGOLNYCH DNI
    active_cases = df_confirmed - df_deaths - df_recovered
    print('\nZADANIE 1 - liczba aktywnych przypadków: ')
    print(active_cases)

    #odrzucenie danych dla mniej niz 100 aktywnych przypadkow zostalo zaimplementowane w task_2
    return active_cases, df_coordinates, df_deaths, df_confirmed, mortality



def task_2(active_cases):

    # ODRZUCENIE DANYCH DLA MNIEJ NIZ 100 AKTYWNYCH PRZYPADKOW
    temp = active_cases.copy()
    active_ov100 = temp[temp >= 100].fillna(0)
    counting = active_ov100.copy()
    counting[active_ov100 > 0] = 1

    active_cases_copy = active_ov100.copy()
    counting_copy = counting.copy()


    # OBLICZANIE 'M'
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


    # OBLICZANIE R
    for i in range(5, len(M.columns.values)):
        R.loc[:, active_cases_copy.columns[i]] = M.loc[:, active_cases_copy.columns[i]] / M_div.loc[:, active_cases_copy.columns[i - 5]]

    return R



def weather(R, df_coordinates):

    weather_max = Dataset('./data/TerraClimate_tmax_2018.nc')
    weather_min = Dataset('./data/TerraClimate_tmin_2018.nc')

    coor_idx_lat_max = pd.DataFrame(weather_max['tmax'][0]).shape[0]
    coor_idx_lon_max = pd.DataFrame(weather_max['tmax'][0]).shape[1]


    # SKALOWANIE WSPOLRZEDNYCH
    rescaled_cols = []
    for col in range(coor_idx_lon_max):
        rescaled_cols.append(int(((180.0 + 180.0) * col / coor_idx_lon_max) - 180))
    rescaled_rows = []
    for row in range(coor_idx_lat_max):
        rescaled_rows.append(int(((90.0 + 90) * row / coor_idx_lat_max) - 90) * (-1))


    # WYZNACZENIE R NA POSZCZEGOLNE MIESIACE
    R.columns = pd.to_datetime(R.columns)
    R_20 = R.groupby([R.columns.year, R.columns.month], axis=1).sum()[2020]
    temp_mean = R_20.copy()
    R_20 = pd.concat([R_20, df_coordinates], axis=1)
    R_20.dropna(inplace=True)


    # WYZNACZANIE SREDNIEJ TEMPERATURY, DLA POSZCZEGOLNYCH WSPOLRZEDNYCH, W KOLEJNYCH MIESIACACH
    for month in range(0, 12):
        df_w_max = pd.DataFrame(weather_max['tmax'][month])
        df_w_min = pd.DataFrame(weather_min['tmin'][month])
        temp_mean_monthly = (df_w_max + df_w_min) / 2
        temp_mean_monthly.index = rescaled_rows
        temp_mean_monthly.columns = rescaled_cols
        for index in range(len(R_20.index.values)-1):
            val = temp_mean_monthly.loc[int(R_20.iloc[index, -2]), int(R_20.iloc[index, -1])].iloc[0, 0]
            temp_mean.iloc[index, month] = val

    return R_20, temp_mean



def hypothesis_task1(R, temp_mean):

    R.drop(columns=['Lat', 'Long'], inplace=True)
    temp_ranged = temp_mean.copy()
    temp_ranged.dropna(inplace=True)

    def bucketing(x):
        x = int( float(x)/ 10.0) + 1
        if x <= 0: x = 0
        elif x >= 4: x = 4
        return x


    # PODZIAL TEMPERATURY NA PRZEDZIALY
    temp_test = temp_ranged.applymap(bucketing)
    R_test = R.div(R.max(axis=1), axis=0)
    R_test.dropna(inplace=True)

    bucket_1 = []
    bucket_2 = []
    bucket_3 = []
    bucket_4 = []
    bucket_5 = []
    buckets = [bucket_1, bucket_2, bucket_3, bucket_4, bucket_5]
    for i in range(len(temp_test.index.values) - 1):
        for j in range(len(temp_test.columns) - 1):
            num = temp_test.iloc[i, j]
            val = R_test.iloc[i, j]
            buckets[num].append(val)


    # TEST ZGODNIE Z PROCEDURA ANOVA
    x = np.concatenate(buckets)
    k2, p = stats.normaltest(x)
    print('\n\n\nTESTOWANIE HIPOTEZ. cz1: ', '\n')
    print('normaltest p_val: ', p)


    # obliczanie wariancji zbiorow
    print('\nwariancje zbiorow:')
    for bucket in buckets:
        print(np.var(bucket))

    f_value, p_value = stats.f_oneway(bucket_1, bucket_2, bucket_3, bucket_4, bucket_5)
    print('\nf_oneway p-val: ', p_value)

    print('\nanaliza post-hoc: \n')
    print(pairwise_tukeyhsd(np.concatenate([bucket_1, bucket_2, bucket_3, bucket_4, bucket_5]),
                            np.concatenate([['bucket_1'] * len(bucket_1),
                                            ['bucket_2'] * len(bucket_2),
                                            ['bucket_3'] * len(bucket_3),
                                            ['bucket_4'] * len(bucket_4),
                                            ['bucket_5'] * len(bucket_5)])))

    print("\n\nWystepuja roznice pomiedzy zbiorami. Temperatura moze miec wplyw na rozprzestrzenianie sie wirusa")





def hypothesis_task2_1(df_deaths, df_confirmed, df_coordinates):

    df_deaths = pd.concat([df_deaths, df_coordinates], axis=1)
    df_deaths.dropna(inplace=True)
    df_confirmed = pd.concat([df_confirmed, df_coordinates], axis=1)
    df_confirmed.dropna(inplace=True)


    # WYZNACZANIE CALKOWITYCH WARTOSCI SMIERCI ORAZ POTWEIRDZONYCH PRZYPADKOW
    df_deaths['sum'] = df_deaths.sum(axis=1)
    df_confirmed['sum'] = df_confirmed.sum(axis=1)


    # WYBIERANIE KRAJOW Z EUROPY
    df_temp0 = df_deaths[(df_deaths['Lat'] < 71)]
    df_temp1 = df_temp0[(df_temp0['Lat'] > 35)]
    df_temp2 = df_temp1[(df_temp1['Long'] < 68)]
    europe_deaths = df_temp2[(df_temp2['Long'] > 9)]['sum'].values

    df_tempC0 = df_deaths[(df_confirmed['Lat'] < 71)]
    df_tempC1 = df_tempC0[(df_tempC0['Lat'] > 35)]
    df_tempC2 = df_tempC1[(df_tempC1['Long'] < 68)]
    europe_comfirmed = df_tempC2[(df_tempC2['Long'] > 9)]['sum'].values
    test_list = list(zip(europe_deaths, europe_comfirmed))


    # TEST CHI2
    chi2, p, dof, expected = chi2_contingency(test_list)
    print('\n\n\nTESTOWANIE HIPOTEZ. cz2: ', '\n')
    print('p-val: ', p)




def hyphotesis_2_2(mortality, df_coordinates):

    mortality = pd.concat([mortality, df_coordinates], axis=1)
    mortality.dropna(inplace=True)

    # WYBIERANIE KRAJOW Z EUROPY
    df_temp0 = mortality[(mortality['Lat'] < 71)]
    df_temp1 = df_temp0[(df_temp0['Lat'] > 35)]
    df_temp2 = df_temp1[(df_temp1['Long'] < 68)]
    europe_mortality_temp = df_temp2[(df_temp2['Long'] > 9)]
    europe_mortality = europe_mortality_temp.drop(columns=['Lat', 'Long'])



    # TEST ZGODNIE Z PROCEDURA ANOVA
    x = np.concatenate(europe_mortality.values)
    k2, p = stats.normaltest(x)
    print('\n\n\nTESTOWANIE HIPOTEZ. cz2_2: ', '\n')
    print('normaltest p_val: ', p)


    # # obliczanie wariancji dla krajow (+odrzucanie)
    print('\nwariancje dla krajow:')
    for country in europe_mortality.index:
        print(country, ': ', np.var(europe_mortality.loc[country].values))

    europe_mortality.drop(index=['Liechtenstein ', 'Malta '], inplace=True)

    f_value, p_value = stats.f_oneway(*[europe_mortality.loc[country].tolist() for country in europe_mortality.index])
    print('\nf_oneway p-val: ', p_value)




if __name__ == '__main__':

    confirmed_df, deaths_df, recovered_df = data_collect()
    active_cases, df_coordinates, df_deaths, df_confirmed, mortality = task_1(confirmed_df, deaths_df, recovered_df)
    R = task_2(active_cases)
    R_20, temp_mean = weather(R, df_coordinates)
    hypothesis_task1(R_20, temp_mean)
    hypothesis_task2_1(df_deaths, df_confirmed, df_coordinates)
    hyphotesis_2_2(mortality, df_coordinates)