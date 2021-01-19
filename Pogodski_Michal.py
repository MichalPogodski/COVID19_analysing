import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

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
            active = df_confirmed.loc[name, date] - df_confirmed.loc[name, (date - DateOffset(days=14))]
            dead = df_deaths.loc[name, date] - df_deaths.loc[name, (date - DateOffset(days=14))]
            active = active - dead
            res = df_confirmed.loc[name, date] - df_deaths.loc[name, date] - active
            df_recovered.loc[name, date] = res


    df_deaths['sum'] = df_deaths.sum(axis=1)
    list_to_drop = df_deaths[df_deaths['sum'] == 0].index
    df_deaths = df_deaths.drop(index=list_to_drop)
    df_recovered = df_recovered.drop(index=list_to_drop)
    df_confirmed = df_confirmed.drop(index=list_to_drop)
    df_deaths = df_deaths.drop(columns=['sum'])

    df_deaths.columns = pd.to_datetime(df_deaths.columns)
    df_deaths_monthly = df_deaths.groupby([df_deaths.columns.year, df_deaths.columns.month], axis=1).sum()
    df_recovered.columns = pd.to_datetime(df_recovered.columns)
    df_recovered_monthly = df_recovered.groupby([df_recovered.columns.year, df_recovered.columns.month], axis=1).sum()
    mortality = df_deaths_monthly / df_recovered_monthly
    print('\nZADANIE 1 - śmiertelność: ')
    print(mortality)

    active_cases = df_confirmed - df_deaths - df_recovered
    print('\nZADANIE 1 - liczba aktywnych przypadków: ')
    print(active_cases)


if __name__ == '__main__':
    confirmed_df, deaths_df, recovered_df = data_collect()
    task_1(confirmed_df, deaths_df, recovered_df)
