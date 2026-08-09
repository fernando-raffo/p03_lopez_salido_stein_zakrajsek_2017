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

import shutil
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
BASE_DIR = config("BASE_DIR")
RAW_DATA_DIR = config("RAW_DATA_DIR")
PROCESSED_DATA_DIR = config("PROCESSED_DATA_DIR")
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
            RAW_DATA_DIR / "fred_data_dictionary.md",
            RAW_DATA_DIR / "fred.csv",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "clean": True,
    }
    yield {
        "name": "Shiller data",
        "doc": "Pull Robert Shiller's stock-market data (CAPE / P/E10)",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_shiller.py",
        ],
        "targets": [
            RAW_DATA_DIR / "shiller_data.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
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
            "python ./src/settings.py",
            "python ./src/pull_greenwood_hanson.py",
        ],
        "targets": [
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            RAW_DATA_DIR / "greenwood_hanson_hys_fisd.parquet",
            RAW_DATA_DIR / "greenwood_hanson_hys_historical.parquet",
            RAW_DATA_DIR / "greenwood_hanson_hys_historical.csv",
            RAW_DATA_DIR / "greenwood_hanson_hys_fisd.csv",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_greenwood_hanson.py",
            MANUAL_DATA_DIR / "greenwood_hanson_hys_historical.csv",
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
            PROCESSED_DATA_DIR / "fred_final_series_annual_readme.md",
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
            PROCESSED_DATA_DIR / "fred_final_series_monthly_readme.md",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/process_fred_data_monthly.py",
            RAW_DATA_DIR / "fred.parquet",
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


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    # notebook_scripts = [
    #    Path(notebook_tasks[notebook]["path"])
    #    for notebook in notebook_tasks.keys()
    # ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        # *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        # "task_dep": [
        #    "run_notebooks",
        # ],
        "clean": True,
    }
