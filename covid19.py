# pylint: disable=C0114,C0115,C0116
from typing import List
import os
import sys
from datetime import datetime
import pandas as pd
from covid19_Param import (
    Covid19Param,
    CSV_PATH,
    COEFFICIENT_FILE,
    CSV_DIR_PATH,
    OUT_DIR_PATH,
    CMD_FILE,
)  # pylint: disable=C0301


def make_params_all(dateframe: pd.DataFrame) -> List[Covid19Param]:
    params = []
    for index in dateframe.index:
        params.append(Covid19Param(index, CSV_PATH, 500))
    return params


def main():
    args = sys.argv
    os.makedirs(CSV_DIR_PATH, exist_ok=True)
    os.makedirs(OUT_DIR_PATH, exist_ok=True)

    os.system(CMD_FILE)

    # CSV読み込み
    Covid19Param.read_corona_csv(CSV_PATH)
    Covid19Param.read_coefficient(COEFFICIENT_FILE)

    # パラメータ作成 (一部抽出した国だけ)
    c_params = [
        Covid19Param(["Japan", "-"], CSV_PATH, 500),
        Covid19Param(["US", "-"], CSV_PATH, 500),
        Covid19Param(["Italy", "-"], CSV_PATH, 500),
        Covid19Param(["China", "Hubei"], CSV_PATH, 500),
        Covid19Param(["Spain", "-"], CSV_PATH, 500),
        Covid19Param(["United Kingdom", "-"], CSV_PATH, 500),
        Covid19Param(["France", "-"], CSV_PATH, 500),
        Covid19Param(["Korea, South", "-"], CSV_PATH, 500),
    ]

    if 1 < len(args):
        if args[1] == "-a":
            # パラメータ作成(データ全部 ← 時間がかかる)
            c_params = make_params_all(Covid19Param.data_df)

    for param in c_params:
        title = param.create_title(param.title)
        title_head = "Covid-19_"
        print("[START]: " + title)
        param.do_predict(title_head, title, OUT_DIR_PATH)
        Covid19Param.add_coefficient(
            title, datetime.today().strftime("%Y%m%d"), param.popt, param.r_squared
        )
        print("[END]: " + title + "\r\n")

    Covid19Param.save_coefficient(COEFFICIENT_FILE)
    return


if __name__ == "__main__":
    main()
