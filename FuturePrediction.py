import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import curve_fit
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def gompertz_curve(x, K, b, c):
    y = K * np.power(b, np.power(math.e, (-1 * c * x)))
    return y

def logistic_curve(x, K, b, c):
    y = K / (1 + b * np.power(math.e, (-1 * c * x)))
    return y

class PredictParam():
    coefficientDf = []
    predictFunc = {'Gompertz' : gompertz_curve, 'Logistic' : logistic_curve}

    def __init__(self, title, csvName, limitTimes, coefficientFile):
        self.title = title
        self.csvName = csvName
        self.limitTimes = limitTimes
        self.iniParams, self.lastR2 = self.__getIniParams(title)
        return

    def __calcCoefficients(self):
        self.popt = {}
        self.pcov = {}
        self.r_squared = {}
        self.r_squared_match = ''
        for key, func in self.predictFunc.items():
            if ((self.iniParams.get(key)[0] == 1) or (np.isnan(self.iniParams.get(key)[0]))):
                self.iniParams[key] = self.__getRandomIniParams(key, self.y_array_count[-1])

            # 前回値で初期化しておく
            popt = self.iniParams[key]
            r_squared = self.lastR2[key]
            pcov = [] #self.lastPcov[key]     # ラストのpcovは覚えていない

            # 収束先が最新実績値より低い場合は現実的ではないので最新実績値としておく
            if (self.iniParams.get(key)[0] < self.y_array_count[-1]):
                self.iniParams[key][0] = self.y_array_count[-1]
            # 前回収束先が上限値より大きい場合は現実的ではないので上限値に丸めておく
            if (self.y_array_count[-1] * self.limitTimes < self.iniParams.get(key)[0]):
                self.iniParams[key][0] = self.y_array_count[-1] * self.limitTimes

            bounds = self.__getBoundParams(key, self.y_array_count[-1], self.limitTimes)
            tmp_popt, tmp_pcov, tmp_r_squared = self.__calcCurveFitting(self.iniParams.get(key), self.y_array_count, func, bounds)
            update_flg = False

            # 初回フィッティング時はとりあえず保持しておく
            if(r_squared == np.nan) and (tmp_r_squared != np.nan):
                popt = tmp_popt
                pcov = tmp_pcov
                r_squared = tmp_r_squared
                update_flg = True

            # 前回精度より今回精度の方が低い or フィッティングできなかった場合はリトライする
            if(((1-r_squared) < (1-tmp_r_squared)) or np.isnan(tmp_r_squared) or tmp_r_squared < 0.8):
                for i in range(50):
                    self.iniParams[key] = self.__getRandomIniParams(key, self.y_array_count[-1])
                    tmp_popt, tmp_pcov, tmp_r_squared = self.__calcCurveFitting(self.iniParams.get(key), self.y_array_count, func, bounds)
                    # よりフィッティング精度の良いものが見つかったら更新して終了
                    if not np.isnan(tmp_r_squared):
                        if(np.isnan(r_squared) or (1-tmp_r_squared) < (1-r_squared)):
                            # print('    ---> Successfully (' + str(i) + ': ' + str(r_squared) + ', ' + str(tmp_r_squared) + ')')
                            popt = tmp_popt
                            pcov = tmp_pcov
                            r_squared = tmp_r_squared
                            update_flg = True
                            if(0.95 < r_squared):
                                break
            else:
                # 今回の方が良かった
                popt = tmp_popt
                pcov = tmp_pcov
                r_squared = tmp_r_squared
                update_flg = True

            # if(update_flg == False):
            #     print('    ---> Use Last Value (No Update: ' + key + ')')

            self.popt[key] = popt
            self.pcov[key] = pcov
            self.r_squared[key] = r_squared

        self.__findMostMatched()
        self.__logoutCoefficients()
        return

    def __findMostMatched(self):
        tmpMin = 1
        for key, value in self.r_squared.items():
            if not np.isnan(value):
                if ((1.0 - value) < tmpMin):
                    tmpKey = key
                    tmpMin = 1.0 - value
        if (tmpMin != 1):
            self.r_squared_match = tmpKey
        return

    def __logoutCoefficients(self):
        for key, value in self.popt.items():
            if (np.isnan(value[0])):
                print('  ' + key + ': ' + 'K = NaN')
            else:
                tmpStr = '  ' + key + ': ' + 'K = '+ str(int(value[0])) + '\tR^2 = ' + str(self.r_squared.get(key))
                if (key == self.r_squared_match):
                    print(tmpStr + ' *')
                else:
                    print(tmpStr)
        return

    def __createGraph(self, title_head, title, out_dir_path):
        days = self.__calcGraphRangeDate(self.y_array_count, 5000, self.popt, self.limitTimes)
        print('  DAYS: ' + str(days))

        # 日付のリスト生成()
        x_array_date, x_array_index = self.createDateArray(self.firstDate, days)
        
        fig = plt.figure()
        for key, func in self.predictFunc.items():
            if(self.popt.get(key)[0] < self.y_array_count[-1]*self.limitTimes):
                labelStr = key + ' ($R^2$='+ str(round(self.r_squared.get(key),3))+')'
                if(key == self.r_squared_match):
                    labelStr = labelStr + ' *'
                plt.plot(x_array_date, func(x_array_index, *self.popt.get(key)), label=labelStr)

        plt.plot(x_array_date[0:len(self.y_array_count)], self.y_array_count, label='Count')
        plt.legend()
        plt.title(title_head + title + ' ' + datetime.today().strftime('%Y%m%d'))
        plt.gcf().autofmt_xdate()
        plt.savefig(out_dir_path + '/' + title_head + title.replace('*', '') + datetime.today().strftime('_%Y%m%d') + '.png')
        plt.close()

        return
    
    def doPredict(self, title_head, title, out_dir_path):
        self.getCountData()
        self.__calcCoefficients()                               # 係数の計算
        self.__createGraph(title_head, title , out_dir_path)    # グラフ作成
        return

    @classmethod
    def __getIniParams(cls, title):
        indexKey = cls.createTitle(title)
        iniParams = {}
        lastR2 = {}
        if (indexKey in cls.coefficientDf.index):
            tmpDf = cls.coefficientDf.fillna(1)
            tmpLastValues = tmpDf.loc[cls.createTitle(title)].tail(1)
            for key, func in cls.predictFunc.items():
                iniParams[key] = [float(tmpLastValues['K('+key[0]+')'].iat[0]),
                                  float(tmpLastValues['b('+key[0]+')'].iat[0]),
                                  float(tmpLastValues['c('+key[0]+')'].iat[0])]
                lastR2[key]    = float(tmpLastValues['R2('+key[0]+')'].iat[0])
        else:
            for key, func in cls.predictFunc.items():
                iniParams[key] = [1.0, 1.0, 1.0]
                lastR2[key]    = np.nan

        return iniParams, lastR2

    @classmethod
    def readCoefficient(cls, file):
        if not (os.path.isfile(file)):
            col = ''
            for key, func in cls.predictFunc.items():
                col = col + ',K(' + key[0] + '),b(' + key[0] + '),c(' + key[0] + '),R2(' + key[0] + ')'
            with open(file, 'a') as f:
                print('Main,Date' + col, file=f)

        df_in = pd.read_csv(file)
        df_in['Date'] = pd.to_datetime(df_in['Date'], format='%Y-%m-%d')
        df_in = df_in.set_index(['Main','Date'])
        cls.coefficientDf = df_in
        return

    @classmethod
    def addCoefficient(cls, Main, date, popt, r_squared):
        tmpDate = pd.to_datetime(date, format='%Y%m%d')

        for key, value in popt.items():
            cls.coefficientDf.loc[(Main, tmpDate), 'K('  + key[0] + ')'] = value[0]
            cls.coefficientDf.loc[(Main, tmpDate), 'b('  + key[0] + ')'] = value[1]
            cls.coefficientDf.loc[(Main, tmpDate), 'c('  + key[0] + ')'] = value[2]
            cls.coefficientDf.loc[(Main, tmpDate), 'R2(' + key[0] + ')'] = r_squared.get(key)
        return

    @classmethod
    def saveCoefficient(cls, file):
        cls.coefficientDf.sort_index(inplace=True)
        cls.coefficientDf.to_csv(file)
        return

    @classmethod
    def __calcGraphRangeDate(cls, y_array_count, range_max, popt, limitTimes):
        tmpDays = []
        for key, func in cls.predictFunc.items():
            tmpDays.append(cls.__calcOptimalRangeDate(y_array_count, range_max, popt.get(key), limitTimes, func))
        days = max(*tmpDays, len(y_array_count))
        return days

    @staticmethod
    def __getRandomIniParams(key, ini_K):
        if (key == 'Gompertz'):
            iniParams = [ini_K, random.random(), random.random()]
        elif (key == 'Logistic'):
            iniParams = [ini_K, random.uniform(0, ini_K), random.random()]
        else:
            iniParams = [ini_K, random.random(), random.random()]
        return iniParams

    @staticmethod
    def __getBoundParams(key, latestCount, limitTimes):
        tmpMin = latestCount
        if (tmpMin < 1.0):
            tmpMin = 1.0
        if (key == 'Gompertz'):
            bounds = ((tmpMin-1, 0.0, 0.0), (tmpMin*limitTimes, 1.0, np.inf))
        elif (key == 'Logistic'):
            bounds = ((tmpMin-1, 0.0, 0.0), (tmpMin*limitTimes, np.inf, np.inf))
        else:
            bounds = ((tmpMin-1, 0.0, 0.0), (tmpMin*limitTimes, np.inf, np.inf))
        return bounds

    @staticmethod
    def __calcCurveFitting(param_ini, y_array_count, curve_func, param_bounds):
        x_array_index = createIndex(len(y_array_count))
        r_squared = np.nan

        try:
            popt, pcov = curve_fit(curve_func, x_array_index, y_array_count, p0=param_ini, bounds=param_bounds)
            if not np.isnan(popt[0]):
                residuals =  y_array_count- curve_func(x_array_index, *popt)
                rss = np.sum(residuals**2)                                  #residual sum of squares = rss
                tss = np.sum((y_array_count - np.mean(y_array_count))**2)   #total sum of squares = tss
                if (tss != 0):
                    r_squared = 1 - (rss / tss)
                else:
                    r_squared = -np.inf

        except ValueError as error:
            popt = [np.nan, np.nan, np.nan]
            pcov = np.nan
            print('   ERROR: ' + str(error))
            print('    param_i' + str(param_ini))
            print('    param_b' + str(param_bounds))
        except RuntimeError as error:
            popt = [np.nan, np.nan, np.nan]
            pcov = np.nan
            #print('   FAIL: ' + str(error) + ' Retry again!')

        return popt, pcov, r_squared

    @staticmethod
    def __calcOptimalRangeDate(y_array_count, range_max, popt, limitTimes, fit_func):
        days = len(y_array_count)
        if not np.isnan(popt[0]):
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
