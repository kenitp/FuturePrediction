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
    def __init__(self, title, csvName, limitTImes):
        self.title = title
        self.csvName = csvName
        self.limitTimes = limitTImes

def gompertz_curve(x, K, b, c):
    y = K * np.power(b, np.power(math.e, (-1 * c * x)))
    return y

def logistic_curve(x, K, b, c):
    y = K / (1 + b * np.power(math.e, (-1 * c * x)))
    return y

def createIndex(size):
   # 0始まりのindexの作成
   return np.arange(0,size, 1)

def createDateArray(firstDate, dayNum):
    #集計開始日
    startDate = datetime(firstDate.year, firstDate.month, firstDate.day, 0, 0, 0)
    # 日付のリスト生成()
    x_array_date = [startDate + timedelta(days=i) for i in range(int(dayNum))]
    # 0始まりのindexの作成
    x_array_index = createIndex(len(x_array_date))

    return x_array_date, x_array_index

def createTitle(titleList):
    title = titleList[0]
    if (len(titleList) == 2):
        if (titleList[1] != '-'):
            title = title + '-' + titleList[1]
    return title

def calcCoefficients(y_array_count):
    x_array_index = createIndex(len(y_array_count))

    param_ini = (y_array_count[-1], 1, 1)
    popt_g, pcov_g = curve_fit(gompertz_curve, x_array_index, y_array_count, p0=param_ini, maxfev=100000000)
    if (np.isnan(popt_g[0])):
        print('Gompertz: ' + 'K = NaN')
    else:
        print('Gompertz: ' + 'K = '+ str(int(popt_g[0])))

    param_ini = (y_array_count[-1], 1, 1)
    popt_l, pcov_l = curve_fit(logistic_curve, x_array_index, y_array_count, p0=param_ini, maxfev=100000000)
    if (np.isnan(popt_l[0])):
        print('Logistic: ' + 'K = NaN')
    else:
        print('Logistic: ' + 'K = '+ str(int(popt_l[0])))

    return popt_g, popt_l

def calcGraphRangeDate(y_array_count, range_max, popt_g, popt_l, limitTimes):
    days_g = 0
    days_l = 0
    if (popt_g[0] != np.nan):
        if (popt_g[0] < y_array_count[-1]*limitTimes):
            for x in range(range_max,0,-1):
                if( (int(gompertz_curve(x, *popt_g)) - int(gompertz_curve(x-1, *popt_g))) > int(gompertz_curve(x, *popt_g) * 0.0001) ):
                    days_g = x
                    break

    if (popt_l[0] != np.nan):
        if (popt_l[0] < y_array_count[-1]*limitTimes):
            for x in range(range_max,0,-1):
                if( (int(logistic_curve(x, *popt_l)) - int(logistic_curve(x-1, *popt_l))) > int(gompertz_curve(x, *popt_l) * 0.0001) ):
                    days_l = x
                    break

    days = max(days_g, days_l, len(y_array_count))
    return days

def createGraph(firstDate, y_array_count, title, limitTimes, popt_g, popt_l, out_dir_path):
    
    days = calcGraphRangeDate(y_array_count, 5000, popt_g, popt_l, limitTimes)
    print('DAYS: ' + str(days))

    # 日付のリスト生成()
    x_array_date, x_array_index = createDateArray(firstDate, days)
    
    fig = plt.figure()
    if(popt_g[0] < y_array_count[-1]*limitTimes):
        plt.plot(x_array_date, gompertz_curve(x_array_index, *popt_g), label='Gompertz')
    if(popt_l[0] < y_array_count[-1]*limitTimes):
        plt.plot(x_array_date, logistic_curve(x_array_index, *popt_l), label='Logistic')
    plt.plot(x_array_date[0:len(y_array_count)], y_array_count, label='Count')
    plt.legend()
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.savefig(out_dir_path + '/' + title + datetime.today().strftime('_%Y%m%d') + '.png')

    with open(out_dir_path + '/Coefficient.csv', 'a') as f:
        print(title + ', ' + datetime.today().strftime('%Y%m%d') + ', ' + 'Gompertz' + ', ' + 'K=' + ', ' + str(popt_g[0]) + ', ' + 'b=' + ', ' + str(popt_g[0]) + ', ' + 'c=' + ', ' + str(popt_g[0]), file=f)
        print(title + ', ' + datetime.today().strftime('%Y%m%d') + ', ' + 'Logistic' + ', ' + 'K=' + ', ' + str(popt_l[0]) + ', ' + 'b=' + ', ' + str(popt_l[0]) + ', ' + 'c=' + ', ' + str(popt_l[0]), file=f)
    return
