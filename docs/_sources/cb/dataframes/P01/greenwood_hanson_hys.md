# Dataframe: `P01:greenwood_hanson_hys` - Greenwood-Hanson High-Yield Share Data (Published Historical, 1926-2008)

## Overview

- **File:** `_data/raw_data/greenwood_hanson_hys_historical.parquet`
- **Source:** Greenwood, Robin, and Samuel G. Hanson (2013), "Issuer Quality and Corporate Bond Returns," *Review of Financial Studies* 26(6), 1483-1525, Table 2. A new vintage of the published series is pulled from the [HBS](https://www.hbs.edu/behavioral-finance-and-financial-stability/Documents/ChartData/LineCharts/InvestorCreditSentiment.xlsx).
- **Pulled by:** `pull_greenwood_hanson.py`
- **Frequency:** Annual, 1926-2008
- **Index:** `year`

## Column Dictionary

| Column | Description |
| --- | --- |
| hy_share | Published Greenwood-Hanson (2013) annual high-yield share (fraction of nonfinancial corporate bond issuance rated below investment grade), from Table 2 of Greenwood and Hanson (2013). |
| ln_hy_share | Natural log of `hy_share`. |
| source | Label identifying the data's provenance; always 'gh2013' for this published historical series. |



## DataFrame Glimpse

```
Rows: 90
Columns: 4
$ hy_share    <f64> 0.2473771
$ ln_hy_share <f64> -1.3968413859595126
$ source      <str> 'gh2013'
$ year        <i32> 2015


```

## Dataframe Manifest

| Dataframe Name                 | Greenwood-Hanson High-Yield Share Data (Published Historical, 1926-2008)                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [greenwood_hanson_hys](../dataframes/P01/greenwood_hanson_hys.md)                                       |
| Sources                        | Greenwood & Hanson (2013), "Issuer Quality and Corporate Bond Returns," Review of Financial Studies 26(6), 1483–1525                                          |
| Providers                      | Greenwood & Hanson, Review of Financial Studies                                        |
| Provider Links                 | https://academic.oup.com/rfs/article-abstract/26/6/1483/1595232                                   |
| Tags                           | Raw Data, Hys, Greenwood, Hanson, High Yield Share                                             |
| Access Types                   | Public                                      |
| How is data pulled?            | HTTP download via Python `requests` (Excel workbook)                                                   |
| Data available up to (min)     | N/A                                                             |
| Data available up to (max)     | N/A                                                             |
| Dataframe Path                 | C:\Users\fraff\OneDrive\Documentos\UChicago\FINM_32900_Full_Stack_Quantitative_Finance\Project\p03_lopez_salido_stein_zakrajsek_2017\_data\raw_data\greenwood_hanson_hys_historical.parquet                                             |


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


