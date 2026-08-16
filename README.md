Lopez-Salido, Stein & Zakrajsek (2017) Replication
=====================================

## About this project

This project replicates key results from:

> Lopez-Salido, D., Stein, J. C., and Zakrajsek, E. (2017), "Credit-Market Sentiment and the Business Cycle." *The Quarterly Journal of Economics*, 132(3): 1373-1426. https://doi.org/10.1093/qje/qjx014

The paper shows that elevated credit-market sentiment in year *t − 2* (proxied by narrow, aggressively priced credit spreads and a high share of junk-bond issuance) forecasts a subsequent widening of credit spreads and a decline in economic activity in years *t* and *t + 1*, using U.S. data from 1929 to 2015. This repo pulls the underlying data (FRED, Greenwood-Hanson credit-spread and issuance data, and Shiller's stock-market data), reconstructs the paper's credit-market sentiment measure via a two-step forecasting regression, and reproduces select tables and figures from the paper, including:

- **Figure 1**: the Baa-Treasury credit spread over time
- **Figure 2**: credit-market sentiment and economic growth
- **Table 1**: forecasting economic growth with credit spreads and stock prices
- **Table 2**:   the two-step regression results linking financial-market sentiment to economic growth

**Extension.** The original paper builds its credit-market sentiment proxy from the spread between Moody's seasoned **Baa**-rated (lowest investment-grade) corporate bond yields and the 10-year Treasury yield. This project extends the replication by rebuilding the same figures and regressions using the spread on Moody's **Aaa**-rated (highest-grade) corporate bonds in place of Baa. The Aaa-based outputs are produced alongside the original Baa-based ones throughout the pipeline.

## Data Sources

| Source | Description |
|--------|-------------|
| FRED | Moody's Aaa/Baa seasoned corporate bond yields, 10-year Treasury yield, 3-month T-bill rate, CPI, population, real GDP, and the NBER recession indicator |
| Robert Shiller's Data Website | Monthly S&P Composite price, dividends, earnings, CPI, the 10-year Treasury rate, and the cyclically adjusted price-earnings ratio (CAPE / P/E10) |
| Greenwood & Hanson (2013), via Harvard Business School | Published historical annual high-yield share of nonfinancial corporate bond issuance, 1926-2008 |
| Mergent FISD via WRDS | Bond-level issuance and rating data used to reconstruct the high-yield share from the early 1980s onward, spliced onto the published Greenwood-Hanson series |

## Quick Start

### 0. Software & Access Prerequisites

1. Conda Package Manager (e.g. [via Anaconda](https://www.anaconda.com/))
2. [Python 3.12 or above](https://www.python.org/)
3. [MacTeX](https://tug.org/mactex/mactex-download.html) or [TeX Live](https://tug.org/texlive/)
4. [WRDS Subscription](https://wrds-www.wharton.upenn.edu/)

### 1. Create & Activate Virtual Environment

You can create a conda environment and all dependencies direcly using the command below if you have the conda package manager:

```bash
conda env create -f environment.yml
conda activate p03_lopez_salido_stein_zakrajsek_2017_env
```

Alternatively, if we also include a `requirements.txt` file to create an environment with alternative package mangers or a simple Python virtual environment:

```bash
conda create -n p03_lopez_salido_stein_zakrajsek_2017_env python=3.12
conda activate p03_lopez_salido_stein_zakrajsek_2017_env
pip install -r requirements.txt
```

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

### 2. Configure WRDS Credentials

Copy .env.example into a new file called .env in the project:

```bash
cp .env.example .env
```

Then edit .env with your WRDS credentials. It should look like:

```bash
WRDS_USERNAME="your_username"
```

### 3. Run the Full Pipeline

```bash
doit
```

### 4. Other commands

#### Unit Tests and Doc Tests

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```

The full `doit` also ends with a `run_tests` task that executes this same suite as its final step, after the data pulls and the Table I / II replications. This means the integration tests that check the replicated coefficients against the published paper run automatically at the end of the pipeline; they skip on their own if the processed data has not been built.

You can build the documentation with:
```
rm ./src/.pytest_cache/README.md
jupyter-book build -W ./
```
Use `del` instead of rm on Windows


#### Setting Environment Variables

You can [export your environment variables](https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file)
from your `.env` files like so, if you wish. This can be done easily in a Linux or Mac terminal with the following command:
```bash
set -a  # automatically export all variables
source .env
set +a
```
On Windows (PowerShell):
```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

## Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code.

```bash
# Auto-fix linting issues (e.g., unused imports, undefined names)
ruff check . --fix

# Format code (consistent style, spacing, line length)
ruff format .

# Sort imports, then fix linting issues, then format
ruff format . && ruff check --select I --fix . && ruff check --fix .
```

- `ruff check --fix` applies safe auto-fixes for linting violations
- `ruff format` formats code similar to Black
- `--select I` targets only import sorting rules (isort-compatible)

## General Directory Structure

```
p03_lopez_salido_stein_zakrajsek_2017/
├── chartbook.toml       # the manifest — configuration for the published ChartBook site
├── dodo.py              # PyDoit task runner — defines and runs the full pipeline
├── environment.yml      # conda environment spec
├── requirements.txt     # pip requirements (alternative to the conda environment)
├── README.md            # this file — also acts as the site's front page
├── .env.example         # sample .env for private paths & WRDS credentials (not tracked in Git)
├── assets/              # hand-drawn figures/logo not generated from code
├── data_manual/         # manually-collected data that can't be recreated (tracked)
├── _data/               # data pulled/processed by the pipeline (gitignored, regenerable)
│   ├── raw_data/        
│   ├── processed_data/          
│   └── data_dictionaries/       
├── _output/             # chart HTML/PDF and tables generated by the pipeline (gitignored, regenerable)
├── docs_src/
│   └── site/            # extra site pages merged into the published site
├── docs/                # built ChartBook/Jupyter Book site (gitignored)
├── reports/             # LaTeX report and bibliography
└── src/                 # code that produces the artifacts: pulls, processing, replication, notebooks, tests
```

### Additional Notes

 - We are using the `doit` Python module as a task runner. It works like `make` and
   the associated `Makefile`s. To rerun the code, install `doit`
   (https://pydoit.org/) and execute the command `doit` from the `src`
   directory. Note that doit is very flexible and can be used to run code
   commands from the command prompt, thus making it suitable for projects that
   use scripts written in multiple different programming languages.

 - We are using the `.env` file as a container for absolute paths that are private
   to each collaborator in the project. You can also use it for private
   credentials, if needed. It should not be tracked in Git.

## Data and Output Storage

We'll often use a separate folder for storing data. Any data in the data folder
can be deleted and recreated by rerunning the PyDoit command (the pulls are in
the dodo.py file). Any data that cannot be automatically recreated should be
stored in the "data_manual" folder. Because of the risk of manually-created data
getting changed or lost, we keep it under version control if I can.
Thus, data in the "_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and
rendered notebooks.
