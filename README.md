# COVID-19 感染拡大予測
 
新型コロナウイルスの感染者数(実績)データから、
* ゴンペルツ曲線
* ロジスティック曲線

にて今後の拡大を予測するプログラム

感染者数の元データは以下より取得しています。

https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv

# Requirement

* Windows 10 
* python 3.11.5
* Rye
* pandas
* scipy
* numpy
* matplotlib
 
# Usage
 
```bash
git clone https://github.com/kenitp/FuturePrediction.git
cd FuturePrediction
rye sync
rye run python covid19.py
```
## Options
-a: All countries in data.

# License
This program is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
