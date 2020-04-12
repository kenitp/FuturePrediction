from covid19_Param import *

cmd_file     = '_GetCmd\getCovid19Data.cmd'
csv_dir_path = '_InData/Covid-19'
csv_path     = csv_dir_path + '/time_series_covid19_confirmed_global.csv'
out_dir_path = './_OutData/Covid-19'

CParams = [ Covid19Param(['Japan'       , '-'    ], csv_path, 500),
            Covid19Param(['US'          , '-'    ], csv_path, 500),
            Covid19Param(['Italy'       , '-'    ], csv_path, 500),
            Covid19Param(['China'       , 'Hubei'], csv_path, 500),
            Covid19Param(['Korea, South', '-'    ], csv_path, 500)]

def main():
    os.makedirs(csv_dir_path, exist_ok=True)
    os.makedirs(out_dir_path, exist_ok=True)
    os.system(cmd_file)

    # CSV読み込み
    Covid19Param.readCoronaCsv(csv_path)

    for param in CParams:
        title = createTitle(param.title)
        title = 'Covid-19_' + title
        print('[START]: ' + title)
        firstDate, y_array_count = param.getCountData(param)
        popt_g, popt_l = calcCoefficients(y_array_count)                            # 係数の計算
        createGraph(firstDate, y_array_count, title , 
                    param.limitTimes, popt_g, popt_l, out_dir_path)                     # グラフ作成
        print('[END]: ' + title + '\r\n')

    return

if __name__ == '__main__':
    main()