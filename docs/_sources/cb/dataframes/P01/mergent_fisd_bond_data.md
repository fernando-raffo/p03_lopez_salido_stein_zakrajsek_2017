# Dataframe: `P01:mergent_fisd_bond_data` - Mergent FISD High-Yield Share Data (Raw FISD Reconstruction, 1980s-present)

## Overview

- **File:** `_data/raw_data/greenwood_hanson_hys_fisd.parquet`
- **Source:** Mergent FISD (Fixed Income Securities Database), via WRDS
- **Pulled by:** `pull_greenwood_hanson.py`
- **Frequency:** Annual, effectively from the early 1980s onward
- **Index:** `year`

## Column Dictionary

| Column | Description |
| --- | --- |
| hy_issuance | Total face amount of nonfinancial U.S. corporate bonds issued in the year that are rated below investment grade (high yield), reconstructed from Mergent FISD via WRDS. |
| total_issuance | Total face amount of all rated nonfinancial U.S. corporate bonds issued in the year, reconstructed from Mergent FISD via WRDS. |
| n_issues | Number of bond issues underlying the year's `total_issuance`; useful for screening out thin early years, e.g. `df.loc[df.n_issues >= 25]`. |
| hy_share | High-yield share for the year, computed as `hy_issuance / total_issuance`. |
| ln_hy_share | Natural log of `hy_share` (NaN when `hy_share` is 0). |



## DataFrame Glimpse

```
Rows: 80
Columns: 6
$ hy_issuance    <f64> 135685362.25
$ total_issuance <f64> 920322501.931
$ n_issues       <i64> 1517
$ hy_share       <f64> 0.14743240762374932
$ ln_hy_share    <f64> -1.914385461630749
$ year           <i32> 2025


```

## Dataframe Manifest

| Dataframe Name                 | Mergent FISD High-Yield Share Data (Raw FISD Reconstruction, 1980s-present)                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [mergent_fisd_bond_data](../dataframes/P01/mergent_fisd_bond_data.md)                                       |
| Sources                        | Mergent FISD                                          |
| Providers                      | WRDS                                        |
| Provider Links                 | https://wrds-www.wharton.upenn.edu/                                   |
| Tags                           | Raw Data, Hys, Fisd, High Yield Share, Wrds, Mergent                                             |
| Access Types                   | WRDS Subscription                                      |
| How is data pulled?            | WRDS Python API                                                   |
| Data available up to (min)     | N/A                                                             |
| Data available up to (max)     | N/A                                                             |
| Dataframe Path                 | C:\Users\fraff\OneDrive\Documentos\UChicago\FINM_32900_Full_Stack_Quantitative_Finance\Project\p03_lopez_salido_stein_zakrajsek_2017\_data\raw_data\greenwood_hanson_hys_fisd.parquet                                             |


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


