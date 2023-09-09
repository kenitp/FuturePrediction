# pylint: disable=C0114,C0115,C0116,C0103
import pandas as pd
from FuturePrediction import PredictParam

CMD_FILE = "_GetCmd\\getCovid19Data.cmd"
CSV_DIR_PATH = "_InData/Covid-19"
CSV_PATH = CSV_DIR_PATH + "/time_series_covid19_confirmed_global.csv"
OUT_DIR_PATH = "./_OutData/Covid-19"
COEFFICIENT_FILE = OUT_DIR_PATH + "/Coefficient.csv"


class Covid19Param(PredictParam):
    @classmethod
    def read_corona_csv(cls, path: str) -> None:
        df_in = pd.read_csv(path)
        df_in.columns.name = "date"
        df_in.drop("Lat", axis=1, inplace=True)
        df_in.drop("Long", axis=1, inplace=True)
        df_in["Province/State"] = df_in["Province/State"].fillna("-")
        df_in = df_in.set_index(["Country/Region", "Province/State"])
        df_in.columns = pd.to_datetime(df_in.columns, format="mixed")
        cls.data_df = df_in
        return
