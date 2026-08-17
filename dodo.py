"""
Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
# Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import glob
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
RAW_DATA_DIR = config("RAW_DATA_DIR")
PROCESSED_DATA_DIR = config("PROCESSED_DATA_DIR")
DATA_DICTIONARY_DIR = config("DATA_DICTIONARY_DIR")
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

# Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"


def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"


def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"


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
        "doc": "Pull Robert Shiller's stock-market data",
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
        "doc": "Build the Greenwood-Hanson high-yield share data",
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
    """Clean the raw FRED pull into the annual and monthly series used throughout the replication"""
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
            "./src/pull_fred.py",
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
            "./src/pull_fred.py",
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
            "./src/latex_format.py",
            "./src/plot_style.py",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
        ],
        "clean": True,
    }


def task_replicate_figure_1():
    """Replicate and Extend LSZ (2017) Figure I"""
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
    """Replicate and Extend LSZ (2017) Table I"""
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
            "./src/helper_functions.py",
            "./src/latex_format.py",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
        ],
        "clean": True,
    }


def task_replicate_table_2():
    """Replicate and Extend LSZ (2017) Table II"""
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
            "./src/helper_functions.py",
            "./src/latex_format.py",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
        ],
        "clean": True,
    }


def task_replicate_figure_2():
    """Replicate and Extend LSZ (2017) Figure II"""
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
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
        ],
        "clean": True,
    }


def task_generate_interactive_charts():
    """Generate interactive (Plotly/HTML) versions of Figures I and II for the ChartBook site"""
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
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            "./src/settings.py",
            "./src/summary_statistics.py",
        ],
        "targets": [],
    },
    "02_replication.ipynb": {
        "path": "./src/02_replication.ipynb",
        "file_dep": [
            "./src/settings.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            "./src/settings.py",
            "./src/replicate_figure_1.py",
            "./src/replicate_table_1.py",
            "./src/helper_functions.py",
            "./src/plot_style.py",
            "./src/replicate_figure_2.py",
            "./src/replicate_table_2.py",
        ],
        "targets": [],
    },
    "03_extension.ipynb": {
        "path": "./src/03_extension.ipynb",
        "file_dep": [
            "./src/settings.py",
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet",
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet",
            PROCESSED_DATA_DIR / "shiller_data_annual.parquet",
            PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet",
            "./src/settings.py",
            "./src/replicate_figure_1.py",
            "./src/replicate_table_1.py",
            "./src/helper_functions.py",
            "./src/plot_style.py",
            "./src/replicate_figure_2.py",
            "./src/replicate_table_2.py",
        ],
        "targets": [],
    },
    "04_case_study.ipynb": {
        "path": "./src/04_case_study.ipynb",
        "file_dep": [
            "./src/settings.py",
            "./src/plot_style.py",
            "./src/replicate_table_2.py",
        ],
        "targets": [
            OUTPUT_DIR / "case_study_covid_oos.pdf",
        ],
    },
}


def task_run_notebooks():
    """Execute each project notebook in place and export it to HTML for the ChartBook site"""
    for notebook in notebook_tasks:
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
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }


def task_compile_latex_report():
    """Compile the LaTeX replication writeup (report.tex) to a final PDF report"""
    return {
        "actions": [
            "python ./src/collect_summary_for_report.py",
            "latexmk -pdf -halt-on-error -cd ./reports/report.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/report.tex",
        ],
        "file_dep": [
            "./reports/report.tex",
            "./reports/references.bib",
            "./src/settings.py",
            "./src/collect_summary_for_report.py",
            OUTPUT_DIR / "summary_statistics_credit_spreads.tex",
            OUTPUT_DIR / "summary_statistics_credit_spreads.pdf",
            OUTPUT_DIR / "summary_statistics_gdp_growth.tex",
            OUTPUT_DIR / "summary_statistics_gdp_growth.pdf",
            OUTPUT_DIR / "summary_statistics_hy_share.tex",
            OUTPUT_DIR / "summary_statistics_hy_share.pdf",
            OUTPUT_DIR / "summary_statistics_cape.tex",
            OUTPUT_DIR / "summary_statistics_cape.pdf",
            OUTPUT_DIR / "table_1_replication.tex",
            OUTPUT_DIR / "table_2_replication.tex",
            OUTPUT_DIR / "figure_1_replication.pdf",
            OUTPUT_DIR / "figure_2_replication.pdf",
            OUTPUT_DIR / "table_1_extended.tex",
            OUTPUT_DIR / "table_2_extended.tex",
            OUTPUT_DIR / "figure_1_extended.pdf",
            OUTPUT_DIR / "figure_2_extended.pdf",
            OUTPUT_DIR / "table_1_aaa_replication.tex",
            OUTPUT_DIR / "table_2_aaa_replication.tex",
            OUTPUT_DIR / "figure_1_aaa_replication.pdf",
            OUTPUT_DIR / "figure_2_aaa_replication.pdf",
            OUTPUT_DIR / "table_1_aaa_extended.tex",
            OUTPUT_DIR / "table_2_aaa_extended.tex",
            OUTPUT_DIR / "figure_1_aaa_extended.pdf",
            OUTPUT_DIR / "figure_2_aaa_extended.pdf",
            OUTPUT_DIR / "case_study_covid_oos.pdf",
        ],
        "targets": ["./reports/report.pdf"],
        "clean": True,
    }


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Build the ChartBook/Sphinx documentation site from the notebooks, data dictionaries, and charts declared in chartbook.toml"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"]) for notebook in notebook_tasks
    ]
    chart_html_targets = [
        OUTPUT_DIR / f"{name}.html"
        for name in [
            "figure_1_replication",
            "figure_1_extended",
            "figure_1_aaa_replication",
            "figure_1_aaa_extended",
            "figure_2_replication",
            "figure_2_extended",
            "figure_2_aaa_replication",
            "figure_2_aaa_extended",
            "summary_statistics_credit_spreads",
            "summary_statistics_gdp_growth",
            "summary_statistics_hy_share",
            "summary_statistics_cape",
        ]
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *glob.glob(str(DATA_DICTIONARY_DIR / "*.md")),
        *glob.glob(str(RAW_DATA_DIR / "*.parquet")),
        *glob.glob(str(PROCESSED_DATA_DIR / "*.parquet")),
        *notebook_scripts,
        *chart_html_targets,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "task_dep": [
            "pull_data",
            "process_fred_data",
            "replicate_table_1",
            "replicate_table_2",
        ],
        "clean": True,
        "verbosity": 2,
    }
