"""
Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import glob
import shutil
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
BASE_DIR = config("BASE_DIR")
RAW_DATA_DIR = config("RAW_DATA_DIR")
PROCESSED_DATA_DIR = config("PROCESSED_DATA_DIR")
DATA_DICTIONARY_DIR = config("DATA_DICTIONARY_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": True,
    }


def task_pull_data():
    """Pull data from FRED, Shiller, and Greenwood-Hanson (high-yield share)"""
    yield {
        "name": "FRED data",
        "doc": "Pull data from FRED",
        "actions": [
            "python ./src/pull_fred.py",
        ],
        "targets": [
            RAW_DATA_DIR / "fred.parquet",
            DATA_DICTIONARY_DIR / "fred_data_dictionary.md",
            RAW_DATA_DIR / "fred.csv",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "clean": True,
    }
    yield {
        "name": "Shiller data",
        "doc": "Pull Robert Shiller's stock-market data (CAPE / P/E10)",
        "actions": [
            "python ./src/pull_shiller.py",
        ],
        "targets": [
            RAW_DATA_DIR / "shiller_data.parquet",
            DATA_DICTIONARY_DIR / "shiller_data_dictionary.md",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            DATA_DICTIONARY_DIR / "shiller_data_annual_dictionary.md",
            RAW_DATA_DIR / "shiller_data.csv",
            PROCESSED_DATA_DIR / "shiller_data_annual.csv",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_shiller.py"],
        "clean": True,
    }
    yield {
        "name": "Greenwood-Hanson high-yield share data",
        "doc": (
            "Build the Greenwood-Hanson high-yield share (needs WRDS/Mergent "
            "FISD, or a raw issuance file in data_manual with GH_HYS_SOURCE=raw)"
        ),
        "actions": [
            "python ./src/pull_greenwood_hanson.py",
        ],
        "targets": [
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            DATA_DICTIONARY_DIR / "greenwood_hanson_hys_dictionary.md",
            RAW_DATA_DIR / "greenwood_hanson_hys_fisd.parquet",
            DATA_DICTIONARY_DIR / "greenwood_hanson_hys_fisd_dictionary.md",
            RAW_DATA_DIR / "greenwood_hanson_hys_historical.parquet",
            DATA_DICTIONARY_DIR / "greenwood_hanson_hys_historical_dictionary.md",
            RAW_DATA_DIR / "greenwood_hanson_hys_historical.csv",
            RAW_DATA_DIR / "greenwood_hanson_hys_fisd.csv",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_greenwood_hanson.py",
        ],
        "clean": True,
    }


def task_process_fred_data():
    """Process FRED data"""
    yield {
        "name": "Clean FRED annual data",
        "doc": "Create clean series required for replication from FRED data (annual)",
        "actions": [
            "python ./src/process_fred_data_annual.py",
        ],
        "targets": [
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            DATA_DICTIONARY_DIR / "fred_final_series_annual_readme.md",
            PROCESSED_DATA_DIR / "fred_final_series_annual.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/process_fred_data_annual.py",
            RAW_DATA_DIR / "fred.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "Clean FRED monthly data",
        "doc": "Create clean series required for replication from FRED data (monthly)",
        "actions": [
            "python ./src/process_fred_data_monthly.py",
        ],
        "targets": [
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            DATA_DICTIONARY_DIR / "fred_final_series_monthly_readme.md",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/process_fred_data_monthly.py",
            RAW_DATA_DIR / "fred.parquet",
        ],
        "clean": True,
    }


def task_summary_statistics():
    """Generate summary statistics tables and graphs for the data"""
    return {
        "actions": ["python ./src/summary_statistics.py"],
        "targets": [
            OUTPUT_DIR / "summary_statistics_credit_spreads.tex",
            OUTPUT_DIR / "summary_statistics_credit_spreads.pdf",
            OUTPUT_DIR / "summary_statistics_credit_spreads.html",
            OUTPUT_DIR / "summary_statistics_gdp_growth.tex",
            OUTPUT_DIR / "summary_statistics_gdp_growth.pdf",
            OUTPUT_DIR / "summary_statistics_gdp_growth.html",
            OUTPUT_DIR / "summary_statistics_hy_share.tex",
            OUTPUT_DIR / "summary_statistics_hy_share.pdf",
            OUTPUT_DIR / "summary_statistics_hy_share.html",
            OUTPUT_DIR / "summary_statistics_cape.tex",
            OUTPUT_DIR / "summary_statistics_cape.pdf",
            OUTPUT_DIR / "summary_statistics_cape.html",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/summary_statistics.py",
            "./src/helper_functions.py",
            "./src/plot_style.py",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
        ],
        "clean": True,
    }


def task_replicate_figure_1():
    """Replicate LSZ (2017) Figure I: Baa- and Aaa-Treasury credit spread, 1925-2015."""
    return {
        "actions": ["python ./src/replicate_figure_1.py"],
        "targets": [
            OUTPUT_DIR / "figure_1_replication.pdf",
            OUTPUT_DIR / "figure_1_extended.pdf",
            OUTPUT_DIR / "figure_1_aaa_replication.pdf",
            OUTPUT_DIR / "figure_1_aaa_extended.pdf",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/replicate_figure_1.py",
            "./src/plot_style.py",
            "./src/plot_style.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
        ],
        "clean": True,
    }


def task_replicate_table_1():
    """Replicate LSZ (2017) Table I (Baa spread, plus an Aaa-spread variant)."""
    return {
        "actions": ["python ./src/replicate_table_1.py"],
        "targets": [
            OUTPUT_DIR / "table_1_replication.tex",
            OUTPUT_DIR / "table_1_extended.tex",
            OUTPUT_DIR / "table_1_aaa_replication.tex",
            OUTPUT_DIR / "table_1_aaa_extended.tex",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/replicate_table_1.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
        ],
        "clean": True,
    }


def task_replicate_table_2():
    """Replicate LSZ (2017) Table II (Baa spread, plus an Aaa-spread variant)."""
    return {
        "actions": ["python ./src/replicate_table_2.py"],
        "targets": [
            OUTPUT_DIR / "table_2_replication.tex",
            OUTPUT_DIR / "table_2_extended.tex",
            OUTPUT_DIR / "table_2_aaa_replication.tex",
            OUTPUT_DIR / "table_2_aaa_extended.tex",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/replicate_table_2.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
        ],
        "clean": True,
    }


def task_replicate_figure_2():
    """Replicate LSZ (2017) Figure II: Credit-market sentiment and economic growth, 1929-2015 (Baa spread, plus an Aaa-spread variant)."""
    return {
        "actions": ["python ./src/replicate_figure_2.py"],
        "targets": [
            OUTPUT_DIR / "figure_2_replication.pdf",
            OUTPUT_DIR / "figure_2_extended.pdf",
            OUTPUT_DIR / "figure_2_aaa_replication.pdf",
            OUTPUT_DIR / "figure_2_aaa_extended.pdf",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/replicate_figure_2.py",
            "./src/plot_style.py",
            "./src/replicate_table_2.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
        ],
        "clean": True,
    }


def task_generate_interactive_charts():
    """Generate interactive (Plotly/HTML) versions of Figures I and II for the ChartBook site."""
    return {
        "actions": ["python ./src/generate_interactive_charts.py"],
        "targets": [
            OUTPUT_DIR / "figure_1_replication.html",
            OUTPUT_DIR / "figure_1_extended.html",
            OUTPUT_DIR / "figure_1_aaa_replication.html",
            OUTPUT_DIR / "figure_1_aaa_extended.html",
            OUTPUT_DIR / "figure_2_replication.html",
            OUTPUT_DIR / "figure_2_extended.html",
            OUTPUT_DIR / "figure_2_aaa_replication.html",
            OUTPUT_DIR / "figure_2_aaa_extended.html",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/generate_interactive_charts.py",
            "./src/plot_style.py",
            "./src/replicate_figure_2.py",
            "./src/replicate_table_2.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
        ],
        "clean": True,
    }


notebook_tasks = {
    "01_summary_statistics.ipynb": {
        "path": "./src/01_summary_statistics.ipynb",
        "file_dep": [
            OUTPUT_DIR / "summary_statistics_credit_spreads.tex",
        ],
        "task_dep": ["summary_statistics"],
        "targets": [],
    },
    "02_replication.ipynb": {
        "path": "./src/02_replication.ipynb",
        "file_dep": [
            OUTPUT_DIR / "table_1_replication.tex",
        ],
        "task_dep": [
            "pull_data",
            "replicate_table_1",
            "replicate_table_2",
            "replicate_figure_1",
            "replicate_figure_2",
        ],
        "targets": [],
    },
    "03_extension.ipynb": {
        "path": "./src/03_extension.ipynb",
        "file_dep": [
            OUTPUT_DIR / "table_1_replication.tex",
        ],
        "task_dep": [
            "pull_data",
            "replicate_table_1",
            "replicate_table_2",
            "replicate_figure_1",
            "replicate_figure_2",
        ],
        "targets": [],
    },
    "04_case_study.ipynb": {
        "path": "./src/04_case_study.ipynb",
        "file_dep": [
            OUTPUT_DIR / "table_1_replication.tex",
        ],
        "task_dep": [
            "pull_data",
            "replicate_table_1",
            "replicate_table_2",
            "replicate_figure_1",
            "replicate_figure_2",
        ],
        "targets": [],
    },
}


def task_run_notebooks():
    """
    Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        notebook_path = Path("./src") / notebook
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_path),
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path, OUTPUT_DIR),
            ],
            "file_dep": [
                notebook_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "task_dep": notebook_tasks[notebook].get("task_dep", []),
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"]) for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *glob.glob("./_data/data_dictionaries/*.md"),
        *glob.glob("./_data/raw_data/*.parquet"),
        *glob.glob("./_data/processed_data/*.parquet"),
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],
        "targets": sphinx_targets,
        "file_dep": file_dep,
        # "task_dep": [
        #    "run_notebooks",
        # ],
        "clean": True,
    }


def task_compile_latex_report():
    """Compile the LaTeX replication writeup (report.tex) to PDF (#32)."""
    return {
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/report.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/report.tex",
        ],
        "file_dep": [
            "./reports/report.tex",
            "./reports/references.bib",
            "./_output/table_1_replication.tex",
            "./_output/table_2_replication.tex",
            "./_output/figure_1_replication.pdf",
            "./_output/figure_2_replication.pdf",
            "./_output/table_1_extended.tex",
            "./_output/table_2_extended.tex",
            "./_output/figure_1_extended.pdf",
            "./_output/figure_2_extended.pdf",
            "./_output/table_1_aaa_replication.tex",
            "./_output/table_2_aaa_replication.tex",
            "./_output/figure_1_aaa_replication.pdf",
            "./_output/figure_2_aaa_replication.pdf",
            "./_output/table_1_aaa_extended.tex",
            "./_output/table_2_aaa_extended.tex",
            "./_output/figure_1_aaa_extended.pdf",
            "./_output/figure_2_aaa_extended.pdf",
            "./_output/case_study_covid_oos.pdf",
        ],
        "targets": ["./reports/report.pdf"],
        "clean": True,
    }


def task_run_tests():
    """Run the pytest suite (unit tests, doctests, and paper-match checks).

    Wired as the final pipeline step, following the cookiecutter_chartbook
    convention of ending `doit` with the tests. It depends on the data pulls
    and the Table I / II replications so that the integration tests, which
    compare the replicated coefficients against the published QJE numbers,
    have their processed-parquet inputs built before they run. The pure unit
    tests and doctests run regardless; the paper-match tests skip on their own
    if the data is somehow absent.
    """
    return {
        "actions": ["pytest --doctest-modules src"],
        "task_dep": [
            "pull_data",
            "process_fred_data",
            "replicate_table_1",
            "replicate_table_2",
        ],
        "verbosity": 2,
    }
