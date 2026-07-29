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
        "clean": [],
    }


def task_pull_fred():
    """Pull FRED data"""
    yield {
        "name": "FRED",
        "doc": "Pull data from FRED",
        "actions": [
            "python ./src/pull_fred.py",
        ],
        "targets": [
            RAW_DATA_DIR / "fred.parquet",
            RAW_DATA_DIR / "fred_data_dictionary.md",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "clean": [],
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
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/process_fred_data_annual.py",
            RAW_DATA_DIR / "fred.parquet",
        ],
        "clean": [],
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
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/process_fred_data_monthly.py",
            RAW_DATA_DIR / "fred.parquet",
        ],
        "clean": [],
    }
