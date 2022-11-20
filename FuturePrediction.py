# pylint: disable=C0114,C0115,C0116,C0103
import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def gompertz_curve(x, K, b, c):
    y = K * np.power(b, np.power(math.e, (-1 * c * x)))
    return y


def logistic_curve(x, K, b, c):
    y = K / (1 + b * np.power(math.e, (-1 * c * x)))
    return y


class PredictParam:
    data_df = None
    coefficient_df = None
    predictFunc = {"Gompertz": gompertz_curve, "Logistic": logistic_curve}

    def __init__(self, title, csv_name: str, limit_times: int):
        self.title = title
        self.csv = csv_name
        self.limit_times = limit_times
        self.ini_params, self.lastR2 = self.__get_ini_params(title)
        self.popt = None
        self.pcov = None
        self.r_squared = None
        self.r_squared_match = None
        self.y_array_count = []
        self.first_date = None
        return

    def __calc_coefficients(self):
        self.popt = {}
        self.pcov = {}
        self.r_squared = {}
        self.r_squared_match = ""
        for key, func in self.predictFunc.items():
            if (self.ini_params.get(key)[0] == 1) or (
                np.isnan(self.ini_params.get(key)[0])
            ):
                self.ini_params[key] = self.__get_random_ini_params(
                    key, self.y_array_count[-1]
                )

            # 前回値で初期化しておく
            popt = self.ini_params[key]
            r_squared = self.lastR2[key]
            pcov = []  # self.lastPcov[key]     # ラストのpcovは覚えていない

            # 収束先が最新実績値より低い場合は現実的ではないので最新実績値としておく
            if self.ini_params.get(key)[0] < self.y_array_count[-1]:
                self.ini_params[key][0] = self.y_array_count[-1]
            # 前回収束先が上限値より大きい場合は現実的ではないので上限値に丸めておく
            if self.y_array_count[-1] * self.limit_times < self.ini_params.get(key)[0]:
                self.ini_params[key][0] = self.y_array_count[-1] * self.limit_times

            bounds = self.__get_bound_params(
                key, self.y_array_count[-1], self.limit_times
            )
            tmp_popt, tmp_pcov, tmp_r_squared = self.__calc_curve_fitting(
                self.ini_params.get(key), self.y_array_count, func, bounds
            )

            # 初回フィッティング時はとりあえず保持しておく
            if (r_squared == np.nan) and (tmp_r_squared != np.nan):
                popt = tmp_popt
                pcov = tmp_pcov
                r_squared = tmp_r_squared

            # 前回精度より今回精度の方が低い or フィッティングできなかった場合はリトライする
            if (
                ((1 - r_squared) < (1 - tmp_r_squared))
                or np.isnan(tmp_r_squared)
                or tmp_r_squared < 0.990
            ):
                for i in range(50):  # pylint: disable=W0612
                    self.ini_params[key] = self.__get_random_ini_params(
                        key, self.y_array_count[-1]
                    )
                    tmp_popt, tmp_pcov, tmp_r_squared = self.__calc_curve_fitting(
                        self.ini_params.get(key), self.y_array_count, func, bounds
                    )
                    # よりフィッティング精度の良いものが見つかったら更新して終了
                    if not np.isnan(tmp_r_squared):
                        if np.isnan(r_squared) or (1 - tmp_r_squared) < (1 - r_squared):
                            popt = tmp_popt
                            pcov = tmp_pcov
                            r_squared = tmp_r_squared
                            if 0.995 < r_squared:
                                break
            else:
                # 今回の方が良かった
                popt = tmp_popt
                pcov = tmp_pcov
                r_squared = tmp_r_squared

            self.popt[key] = popt
            self.pcov[key] = pcov
            self.r_squared[key] = tmp_r_squared

        self.__find_most_matched()
        self.__logout_coefficients()
        return

    def __find_most_matched(self):
        tmp_min = 1
        for key, value in self.r_squared.items():
            if not np.isnan(value):
                if (1.0 - value) < tmp_min:
                    tmp_key = key
                    tmp_min = 1.0 - value
        if tmp_min != 1:
            self.r_squared_match = tmp_key
        return

    def __logout_coefficients(self):
        for key, value in self.popt.items():
            if np.isnan(value[0]):
                print("  " + key + ": " + "K = NaN")
            else:
                tmp_str = (
                    "  "
                    + key
                    + ": "
                    + "K = "
                    + str(int(value[0]))
                    + "\tR^2 = "
                    + str(self.r_squared.get(key))
                )
                if key == self.r_squared_match:
                    print(tmp_str + " *")
                else:
                    print(tmp_str)
        return

    def __create_graph(self, title_head, title, out_dir_path):
        days = self.__calc_graph_range_date(
            self.y_array_count, 5000, self.popt, self.limit_times
        )
        print("  DAYS: " + str(days))

        # 日付のリスト生成()
        x_array_date, x_array_index = self.create_date_array(self.first_date, days)

        plt.figure()
        for key, func in self.predictFunc.items():
            if self.popt.get(key)[0] < self.y_array_count[-1] * self.limit_times:
                label_str = (
                    key + " ($R^2$=" + str(round(self.r_squared.get(key), 3)) + ")"
                )
                if key == self.r_squared_match:
                    label_str = label_str + " *"
                plt.plot(
                    x_array_date,
                    func(x_array_index, *self.popt.get(key)),
                    label=label_str,
                )

        plt.plot(
            x_array_date[0 : len(self.y_array_count)], self.y_array_count, label="Count"
        )
        plt.legend()
        plt.title(title_head + title + " " + datetime.today().strftime("%Y%m%d"))
        plt.gcf().autofmt_xdate()
        plt.savefig(
            out_dir_path
            + "/"
            + title_head
            + title.replace("*", "")
            + datetime.today().strftime("_%Y%m%d")
            + ".png"
        )
        plt.close()

        return

    def __get_count_data(self):
        self.y_array_count = self.data_df.loc[self.title[0], self.title[1]]  # 実績積み上げ値取得
        self.first_date = self.data_df.columns[0]
        return

    def do_predict(self, title_head, title, out_dir_path):
        self.__get_count_data()
        self.__calc_coefficients()  # 係数の計算
        self.__create_graph(title_head, title, out_dir_path)  # グラフ作成
        return

    @classmethod
    def __get_ini_params(cls, title):
        idx_key = cls.create_title(title)
        ini_params = {}
        last_r2 = {}
        if idx_key in cls.coefficient_df.index:
            tmp_df = cls.coefficient_df.fillna(1)
            tmp_last_values = tmp_df.loc[cls.create_title(title)].tail(1)
            for key, func in cls.predictFunc.items():  # pylint: disable=W0612
                ini_params[key] = [
                    float(tmp_last_values["K(" + key[0] + ")"].iat[0]),
                    float(tmp_last_values["b(" + key[0] + ")"].iat[0]),
                    float(tmp_last_values["c(" + key[0] + ")"].iat[0]),
                ]
                last_r2[key] = float(tmp_last_values["R2(" + key[0] + ")"].iat[0])
        else:
            for key, func in cls.predictFunc.items():
                ini_params[key] = [1.0, 1.0, 1.0]
                last_r2[key] = np.nan

        return ini_params, last_r2

    @classmethod
    def read_coefficient(cls, file):
        if not os.path.isfile(file):
            col = ""
            for key, func in cls.predictFunc.items():  # pylint: disable=W0612
                col = (
                    col
                    + ",K("
                    + key[0]
                    + "),b("
                    + key[0]
                    + "),c("
                    + key[0]
                    + "),R2("
                    + key[0]
                    + ")"
                )
            with open(file, "a") as f:
                print("Main,Date" + col, file=f)

        df_in = pd.read_csv(file)
        df_in["Date"] = pd.to_datetime(df_in["Date"], format="%Y-%m-%d")
        df_in = df_in.set_index(["Main", "Date"])
        cls.coefficient_df = df_in
        return

    @classmethod
    def add_coefficient(cls, main, date, popt, r_squared):
        tmp_date = pd.to_datetime(date, format="%Y%m%d")

        for key, value in popt.items():
            cls.coefficient_df.loc[(main, tmp_date), "K(" + key[0] + ")"] = value[0]
            cls.coefficient_df.loc[(main, tmp_date), "b(" + key[0] + ")"] = value[1]
            cls.coefficient_df.loc[(main, tmp_date), "c(" + key[0] + ")"] = value[2]
            cls.coefficient_df.loc[
                (main, tmp_date), "R2(" + key[0] + ")"
            ] = r_squared.get(key)
        return

    @classmethod
    def save_coefficient(cls, file):
        cls.coefficient_df.sort_index(inplace=True)
        cls.coefficient_df.to_csv(file)
        return

    @classmethod
    def __calc_graph_range_date(cls, y_array_count, range_max, popt, limit_times):
        tmpDays = []
        for key, func in cls.predictFunc.items():
            tmpDays.append(
                cls.__calc_optimal_range_date(
                    y_array_count, range_max, popt.get(key), limit_times, func
                )
            )
        days = max(*tmpDays, len(y_array_count))
        return days

    @staticmethod
    def __get_random_ini_params(key, ini_K):  # pylint: disable=C0103
        if key == "Gompertz":
            ini_params = [ini_K, random.random(), random.random()]
        elif key == "Logistic":
            ini_params = [ini_K, random.uniform(0, ini_K), random.random()]
        else:
            ini_params = [ini_K, random.random(), random.random()]
        return ini_params

    @staticmethod
    def __get_bound_params(key, latest_cnt, limit_times):
        tmp_min = latest_cnt
        if tmp_min < 1.0:
            tmp_min = 1.0
        if key == "Gompertz":
            bounds = ((tmp_min - 1, 0.0, 0.0), (tmp_min * limit_times, 1.0, np.inf))
        elif key == "Logistic":
            bounds = ((tmp_min - 1, 0.0, 0.0), (tmp_min * limit_times, np.inf, np.inf))
        else:
            bounds = ((tmp_min - 1, 0.0, 0.0), (tmp_min * limit_times, np.inf, np.inf))
        return bounds

    @staticmethod
    def __calc_curve_fitting(param_ini, y_arr_cnt, func, param_bounds):
        x_arr_idx = create_index(len(y_arr_cnt))
        r_squared = np.nan

        try:
            popt, pcov = curve_fit(
                func, x_arr_idx, y_arr_cnt, p0=param_ini, bounds=param_bounds
            )
            if not np.isnan(popt[0]):
                residuals = y_arr_cnt - func(x_arr_idx, *popt)
                rss = np.sum(residuals**2)  # residual sum of squares = rss
                tss = np.sum(
                    (y_arr_cnt - np.mean(y_arr_cnt)) ** 2
                )  # total sum of squares = tss
                if tss != 0:
                    r_squared = 1 - (rss / tss)
                else:
                    r_squared = -np.inf

        except ValueError as error:
            popt = [np.nan, np.nan, np.nan]
            pcov = np.nan
            print("   ERROR: " + str(error))
            print("    param_i" + str(param_ini))
            print("    param_b" + str(param_bounds))
        except RuntimeError as error:  # pylint: disable=W0612
            popt = [np.nan, np.nan, np.nan]
            pcov = np.nan
            # print('   FAIL: ' + str(error) + ' Retry again!')

        return popt, pcov, r_squared

    @staticmethod
    def __calc_optimal_range_date(y_array_count, range_max, popt, limit_times, func):
        days = len(y_array_count)
        if not np.isnan(popt[0]):
            if popt[0] < y_array_count[-1] * limit_times:
                for x in range(range_max, 0, -1):  # pylint: disable=C0103
                    if (int(func(x, *popt)) - int(func(x - 1, *popt))) > int(
                        func(x, *popt) * 0.0001
                    ):
                        days = x
                        break
        return days

    @staticmethod
    def create_date_array(first_date, day_num):
        # 集計開始日
        start_date = datetime(
            first_date.year, first_date.month, first_date.day, 0, 0, 0
        )
        # 日付のリスト生成()
        x_array_date = [start_date + timedelta(days=i) for i in range(int(day_num))]
        # 0始まりのindexの作成
        x_array_index = create_index(len(x_array_date))

        return x_array_date, x_array_index

    @staticmethod
    def create_title(title_list):
        title = title_list[0]
        if len(title_list) == 2:
            if title_list[1] != "-":
                title = title + "-" + title_list[1]
        return title


def create_index(size):
    # 0始まりのindexの作成
    return np.arange(0, size, 1)
