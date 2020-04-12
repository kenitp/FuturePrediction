import pandas as pd
from datetime import datetime
from FuturePrediction import *

cmd_file     = '_GetCmd\getCovid19Data.cmd'
csv_dir_path = '_InData/Covid-19'
csv_path     = csv_dir_path + '/time_series_covid19_confirmed_global.csv'
out_dir_path = './_OutData/Covid-19'

class Covid19Param(PredictParam):
    df = []
    @classmethod
    def readCoronaCsv(cls, path):
        df_in = pd.read_csv(path)
        df_in.columns.name = 'date'
        df_in = df_in.drop('Lat', axis=1)
        df_in = df_in.drop('Long', axis=1)
        df_in['Province/State'] = df_in['Province/State'].fillna('-')
        df_in = df_in.set_index(['Country/Region','Province/State'])
        df_in.columns = pd.to_datetime(df_in.columns)
        cls.df = df_in
        return

    @classmethod
    def getCountData(cls, param):
        y_array_count = cls.df.loc[param.title[0], param.title[1]]                    # 実績積み上げ値取得
        firstDate = cls.df.columns[0]
        return firstDate, y_array_count

