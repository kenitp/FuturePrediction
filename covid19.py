from covid19_Param import *

cmd_file     = '_GetCmd\getCovid19Data.cmd'
csv_dir_path = '_InData/Covid-19'
csv_path     = csv_dir_path + '/time_series_covid19_confirmed_global.csv'
out_dir_path = './_OutData/Covid-19'
coefficientFile = out_dir_path + '/Coefficient.csv'

def makeParamsAll(df):
    params = []
    for index in df.index:
        params.append(Covid19Param(index, csv_path, 500, coefficientFile))
    return params

def main():
    os.makedirs(csv_dir_path, exist_ok=True)
    os.makedirs(out_dir_path, exist_ok=True)

    os.system(cmd_file)

    # CSV読み込み
    Covid19Param.readCoronaCsv(csv_path)
    Covid19Param.readCoefficient(coefficientFile)

    # パラメータ作成 (一部抽出した国だけ)
    CParams = [ Covid19Param(['Japan'         , '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['US'            , '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['Italy'         , '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['China'         , 'Hubei'], csv_path, 500, coefficientFile),
                Covid19Param(['Spain'         , '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['United Kingdom', '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['France'        , '-'    ], csv_path, 500, coefficientFile),
                Covid19Param(['Korea, South'  , '-'    ], csv_path, 500, coefficientFile)]

    # # パラメータ作成(データ全部 ← 時間がかかる)
    # CParams = makeParamsAll(Covid19Param.df)

    for param in CParams:
        title = createTitle(param.title)
        title_head = 'Covid-19_'
        print('[START]: ' + title)
        firstDate, y_array_count = param.getCountData(param)
        popt_g, popt_l = calcCoefficients(param, y_array_count)               # 係数の計算
        createGraph(firstDate, y_array_count, title_head, title , 
                    param.limitTimes, popt_g, popt_l, out_dir_path)    # グラフ作成
        Covid19Param.addCoefficient(title, datetime.today().strftime('%Y%m%d'), popt_g, popt_l)
        print('[END]: ' + title + '\r\n')

    Covid19Param.saveCoefficient(coefficientFile)
    return

if __name__ == '__main__':
    main()