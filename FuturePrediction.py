import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class PredictParam():
    coefficientDf = []
    def __init__(self, title, csvName, limitTimes, coefficientFile):
        self.title = title
        self.csvName = csvName
        self.limitTimes = limitTimes
        self.iniParam_g, self.iniParam_l = self.__getIniParams(title)
        return

    def __calcCoefficients(self):
        self.popt_g, self.pcov_g = self.__calcCurveFitting(self.iniParam_g, self.y_array_count, gompertz_curve)
        if (np.isnan(self.popt_g[0])):
            print('Gompertz: ' + 'K = NaN')
        else:
            print('Gompertz: ' + 'K = '+ str(int(self.popt_g[0])))

        self.popt_l, self.pcov_l = self.__calcCurveFitting(self.iniParam_l, self.y_array_count, logistic_curve)
        if (np.isnan(self.popt_l[0])):
            print('Logistic: ' + 'K = NaN')
        else:
            print('Logistic: ' + 'K = '+ str(int(self.popt_l[0])))

        return

    def __createGraph(self, title_head, title, out_dir_path):
        days = self.__calcGraphRangeDate(self.y_array_count, 5000, self.popt_g, self.popt_l, self.limitTimes)

        # 日付のリスト生成()
        x_array_date, x_array_index = self.createDateArray(self.firstDate, days)
        
        fig = plt.figure()
        if(self.popt_g[0] < self.y_array_count[-1]*self.limitTimes):
            plt.plot(x_array_date, gompertz_curve(x_array_index, *self.popt_g), label='Gompertz')
        if(self.popt_l[0] < self.y_array_count[-1]*self.limitTimes):
            plt.plot(x_array_date, logistic_curve(x_array_index, *self.popt_l), label='Logistic')
        plt.plot(x_array_date[0:len(self.y_array_count)], self.y_array_count, label='Count')
        plt.legend()
        plt.title(title_head + title + ' ' + datetime.today().strftime('%Y%m%d'))
        plt.gcf().autofmt_xdate()
        plt.savefig(out_dir_path + '/' + title_head + title.replace('*', '') + datetime.today().strftime('_%Y%m%d') + '.png')
        plt.close()

        return
    
    def doPredict(self, title_head, title, out_dir_path):
        self.getCountData()
        self.__calcCoefficients()                             # 係数の計算
        self.__createGraph(title_head, title , out_dir_path)    # グラフ作成
        return

    @classmethod
    def __getIniParams(cls, title):
        key = cls.createTitle(title)
        if (key in cls.coefficientDf.index):
            tmpDf = cls.coefficientDf.fillna(1)
            tmpLastValues = tmpDf.loc[cls.createTitle(title)].tail(1)
            iniParam_g = [float(tmpLastValues['K(G)'].iat[0]),
                            float(tmpLastValues['b(G)'].iat[0]),
                            float(tmpLastValues['c(G)'].iat[0])]
            iniParam_l = [float(tmpLastValues['K(L)'].iat[0]),
                            float(tmpLastValues['b(L)'].iat[0]),
                            float(tmpLastValues['c(L)'].iat[0])]
        else:
            iniParam_g = [1.0, 1.0, 1.0]
            iniParam_l = [1.0, 1.0, 1.0]

        return iniParam_g, iniParam_l

    @classmethod
    def readCoefficient(cls, file):
        if not (os.path.isfile(file)):
            with open(file, 'a') as f:
                print('Main,Date,K(G),b(G),c(G),K(L),b(L),c(L)', file=f)

        df_in = pd.read_csv(file)
        df_in['Date'] = pd.to_datetime(df_in['Date'], format='%Y-%m-%d')
        df_in = df_in.set_index(['Main','Date'])
        cls.coefficientDf = df_in
        return

    @classmethod
    def addCoefficient(cls, Main, date, popt_g, popt_l):
        tmpDate = pd.to_datetime(date, format='%Y%m%d')
        cls.coefficientDf.loc[(Main, tmpDate), 'K(G)'] = popt_g[0]
        cls.coefficientDf.loc[(Main, tmpDate), 'b(G)'] = popt_g[1]
        cls.coefficientDf.loc[(Main, tmpDate), 'c(G)'] = popt_g[2]
        cls.coefficientDf.loc[(Main, tmpDate), 'K(L)'] = popt_l[0]
        cls.coefficientDf.loc[(Main, tmpDate), 'b(L)'] = popt_l[1]
        cls.coefficientDf.loc[(Main, tmpDate), 'c(L)'] = popt_l[2]
        return

    @classmethod
    def saveCoefficient(cls, file):
        cls.coefficientDf.sort_index(inplace=True)
        cls.coefficientDf.to_csv(file)
        return

    @classmethod
    def __calcGraphRangeDate(cls, y_array_count, range_max, popt_g, popt_l, limitTimes):
        days_g = cls.__calcOptimalRangeDate(y_array_count, range_max, popt_g, limitTimes, gompertz_curve)
        days_l = cls.__calcOptimalRangeDate(y_array_count, range_max, popt_l, limitTimes, logistic_curve)
        
        days = max(days_g, days_l, len(y_array_count))
        return days

    @staticmethod
    def __calcCurveFitting(param_ini, y_array_count, curve_func):
        x_array_index = createIndex(len(y_array_count))
        if (param_ini[0] == 1):
            param_ini[0] = y_array_count[-1]
        popt, pcov = curve_fit(curve_func, x_array_index, y_array_count, p0=param_ini, maxfev=100000000)
        return popt, pcov

    @staticmethod
    def __calcOptimalRangeDate(y_array_count, range_max, popt, limitTimes, fit_func):
        days = range_max
        if (popt[0] != np.nan):
            if (popt[0] < y_array_count[-1]*limitTimes):
                for x in range(range_max,0,-1):
                    if( (int(fit_func(x, *popt)) - int(fit_func(x-1, *popt))) > int(fit_func(x, *popt) * 0.0001) ):
                        days = x
                        break
        return days

    @staticmethod
    def createDateArray(firstDate, dayNum):
        #集計開始日
        startDate = datetime(firstDate.year, firstDate.month, firstDate.day, 0, 0, 0)
        # 日付のリスト生成()
        x_array_date = [startDate + timedelta(days=i) for i in range(int(dayNum))]
        # 0始まりのindexの作成
        x_array_index = createIndex(len(x_array_date))

        return x_array_date, x_array_index

    @staticmethod
    def createTitle(titleList):
        title = titleList[0]
        if (len(titleList) == 2):
            if (titleList[1] != '-'):
                title = title + '-' + titleList[1]
        return title

def createIndex(size):
    # 0始まりのindexの作成
    return np.arange(0,size, 1)

def gompertz_curve(x, K, b, c):
    y = K * np.power(b, np.power(math.e, (-1 * c * x)))
    return y

def logistic_curve(x, K, b, c):
    y = K / (1 + b * np.power(math.e, (-1 * c * x)))
    return y
