# Dataframe: `P01:fred_macroeconomic_variables` - FRED Macroeconomic Data

## Overview

- **File:** `_data/raw_data/fred.parquet`
- **Source:** [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/), St. Louis Fed
- **Pulled by:** `pull_fred.py`, via `pandas_datareader.data.DataReader`
- **Frequency:** Mixed (daily/monthly/quarterly/annual, one column per series)
- **Index:** `DATE`
## Column Dictionary

| Column (FRED Series ID) | Description |
| --- | --- |
| AAA | Moody's Seasoned Aaa Corporate Bond Yield (monthly) |
| BAA | Moody's Seasoned Baa Corporate Bond Yield (monthly) |
| GS10 | Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (monthly) |
| M1333BUSM156NNBR | Yield on Long-Term United States Bonds for United States (monthly, 1942-1967) |
| M1333AUSM156NNBR | Yield on Long-Term United States Bonds for United States (monthly, 1919-1944) |
| M1329AUSM193NNBR | Yields on Short-Term United States Securities, Three-Six Month Treasury Notes and Certificates, Three Month Treasury Bills (monthly, 1920-1934) |
| TB3MS | 3-Month Treasury Bill Secondary Market Rate, Discount Basis (monthly) |
| CPIAUCNS | Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (monthly) |
| B230RC0Q173SBEA | Population (quarterly) |
| POPH | National Population (annual) |
| GDPC1 | Real Gross Domestic Product (quarterly) |
| GDPCA | Real Gross Domestic Product (annual) |
| USREC | NBER based Recession Indicators for the United States from the Period following the Peak through the Trough (monthly) |



## DataFrame Glimpse

```
Rows: 1212
Columns: 14
$ AAA                       <f64> 5.31
$ BAA                       <f64> 5.9
$ GS10                      <f64> 4.14
$ M1333BUSM156NNBR          <f64> null
$ M1333AUSM156NNBR          <f64> null
$ M1329AUSM193NNBR          <f64> null
$ TB3MS                     <f64> 3.59
$ CPIAUCNS                  <f64> 324.054
$ B230RC0Q173SBEA           <f64> null
$ POPH                      <f64> null
$ GDPC1                     <f64> null
$ GDPCA                     <f64> null
$ USREC                     <i64> 0
$ DATE             <datetime[ns]> 2025-12-01 00:00:00


```

## Dataframe Manifest

| Dataframe Name                 | FRED Macroeconomic Data                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [fred_macroeconomic_variables](../dataframes/P01/fred_macroeconomic_variables.md)                                       |
| Sources                        | FRED, Office of Financial Research                                          |
| Providers                      | FRED, Office of Financial Research                                        |
| Provider Links                 | https://fred.stlouisfed.org/                                   |
| Tags                           | Raw Data, Macroeconomic Data, Fred                                             |
| Access Types                   | Public                                      |
| How is data pulled?            | Web API via Python `pandas_datareader.data.DataReader`                                                   |
| Data available up to (min)     | 1934-03-01 00:00:00                                                             |
| Data available up to (max)     | 2025-12-01 00:00:00                                                             |
| Dataframe Path                 | C:\Users\fraff\OneDrive\Documentos\UChicago\FINM_32900_Full_Stack_Quantitative_Finance\Project\p03_lopez_salido_stein_zakrajsek_2017\_data\raw_data\fred.parquet                                             |


**Linked Charts:**

- None


## Pipeline Manifest

| Pipeline Name                   | Credit-Market Sentiment and the Business Cycle                       |
|---------------------------------|--------------------------------------------------------|
| Pipeline ID                     | [P01](../../../index.md)              |
| Maintainer                      | Fernando Raffo, Bangjie Xu               |
| Contributors                    | Fernando Raffo, Bangjie Xu |
| Repository                     | https://github.com/fernando-raffo/p03_lopez_salido_stein_zakrajsek_2017                  |
| Pipeline Web Page               | <a href="file://C:/Users/fraff/OneDrive/Documentos/UChicago/FINM_32900_Full_Stack_Quantitative_Finance/Project/p03_lopez_salido_stein_zakrajsek_2017/docs/index.html">Pipeline Web Page      |
| Date of Last Code Update        | 2026-08-19 22:04:54           |
| OS Compatibility                | Windows, Linux, MacOS |
| Linked Dataframes               |  [P01:fred_macroeconomic_variables](../../dataframes/P01/fred_macroeconomic_variables.md)<br>  [P01:shiller_market_variables](../../dataframes/P01/shiller_market_variables.md)<br>  [P01:greenwood_hanson_hys](../../dataframes/P01/greenwood_hanson_hys.md)<br>  [P01:mergent_fisd_bond_data](../../dataframes/P01/mergent_fisd_bond_data.md)<br>  [P01:fred_processed_monthly_series](../../dataframes/P01/fred_processed_monthly_series.md)<br>  [P01:fred_processed_annual_series](../../dataframes/P01/fred_processed_annual_series.md)<br>  [P01:shiller_processed_annual_series](../../dataframes/P01/shiller_processed_annual_series.md)<br>  [P01:greenwood_hanson_hys_processed_series](../../dataframes/P01/greenwood_hanson_hys_processed_series.md)<br>  |


