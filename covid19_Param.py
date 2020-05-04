import pandas as pd
from datetime import datetime
from FuturePrediction import *

cmd_file        = '_GetCmd\getCovid19Data.cmd'
csv_dir_path    = '_InData/Covid-19'
csv_path        = csv_dir_path + '/time_series_covid19_confirmed_global.csv'
out_dir_path    = './_OutData/Covid-19'
coefficientFile = out_dir_path + '/Coefficient.csv'

class Covid19Param(PredictParam):
    df = []

    def getCountData(self):
        self.y_array_count = self.df.loc[self.title[0], self.title[1]]                    # 実績積み上げ値取得
        self.firstDate = self.df.columns[0]
        return

    @classmethod
    def readCoronaCsv(cls, path):
        df_in = pd.read_csv(path)
        df_in.columns.name = 'date'
        df_in.drop('Lat', axis=1, inplace=True)
        df_in.drop('Long', axis=1, inplace=True)
        df_in['Province/State'] = df_in['Province/State'].fillna('-')
        df_in = df_in.set_index(['Country/Region','Province/State'])
        df_in.columns = pd.to_datetime(df_in.columns)
        cls.df = df_in
        return

