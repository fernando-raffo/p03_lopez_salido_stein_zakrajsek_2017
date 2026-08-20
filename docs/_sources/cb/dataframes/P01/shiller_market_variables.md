# Dataframe: `P01:shiller_market_variables` - Robert Shiller's Stock Market Data

## Overview

- **File:** `_data/raw_data/shiller_data.parquet`
- **Source:** [Robert Shiller's data website](https://shillerdata.com/) (the `ie_data.xls` workbook, `Data` sheet)
- **Pulled by:** `pull_shiller.py`
- **Frequency:** Monthly, from 1871-01 onward
- **Index:** `date`
## Column Dictionary

| Column | Description |
| --- | --- |
| sp500_price | S&P Composite (S&P 500 predecessor) nominal price index, monthly. |
| dividend | S&P Composite nominal dividend, monthly, as reported by Shiller. |
| earnings | S&P Composite nominal earnings, monthly, as reported by Shiller. |
| cpi | Consumer Price Index (CPI-U), monthly, as reported in Shiller's data set; used to construct the real (inflation-adjusted) series. |
| gs10 | 10-year U.S. Treasury (long-term government bond) yield, monthly. |
| real_price | S&P Composite price, deflated to real (CPI-adjusted) terms by Shiller. |
| real_dividend | S&P Composite dividend, deflated to real (CPI-adjusted) terms by Shiller. |
| real_earnings | S&P Composite earnings, deflated to real (CPI-adjusted) terms by Shiller. |
| pe10 | Shiller's cyclically adjusted price-earnings ratio (CAPE / P/E10): real price divided by the 10-year moving average of real earnings. |



## DataFrame Glimpse

```
Rows: 1212
Columns: 10
$ sp500_price            <f64> 6853.025454545453
$ dividend               <f64> 79.52
$ earnings               <f64> 240.634
$ cpi                    <f64> 324.054
$ gs10                   <f64> 4.14
$ real_price             <f64> 7043.773219867393
$ real_dividend          <f64> 81.73336727829312
$ real_earnings          <f64> 247.33182974905418
$ pe10                   <f64> 39.58164099324255
$ date          <datetime[ns]> 2025-12-01 00:00:00


```

## Dataframe Manifest

| Dataframe Name                 | Robert Shiller's Stock Market Data                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [shiller_market_variables](../dataframes/P01/shiller_market_variables.md)                                       |
| Sources                        | Robert Shiller's Data Website                                          |
| Providers                      | Robert Shiller                                        |
| Provider Links                 | https://shillerdata.com/                                   |
| Tags                           | Raw Data, Stock Market, Shiller                                             |
| Access Types                   | Public                                      |
| How is data pulled?            | HTTP download via Python `requests`                                                   |
| Data available up to (min)     | N/A                                                             |
| Data available up to (max)     | N/A                                                             |
| Dataframe Path                 | C:\Users\fraff\OneDrive\Documentos\UChicago\FINM_32900_Full_Stack_Quantitative_Finance\Project\p03_lopez_salido_stein_zakrajsek_2017\_data\raw_data\shiller_data.parquet                                             |


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


