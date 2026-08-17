"""
Load project configurations from .env files or from the command line.

Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `settings.py` is run on its own, it will create the appropriate
directories.
"""

import sys
from datetime import datetime
from pathlib import Path
from platform import system

from decouple import config as _config


def find_all_caps_cli_vars(argv=sys.argv):
    """
    Find all command line arguments that are all caps.

    Find all command line arguments that are all caps and defined
    with a long option, for example, --DATA_DIR or --MANUAL_DATA_DIR.
    When that option is found, the value of the option is returned.
    """
    result = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        # Handle --VAR=value format
        if arg.startswith("--") and "=" in arg and arg[2:].split("=")[0].isupper():
            var_name, value = arg[2:].split("=", 1)
            result[var_name] = value
        # Handle --VAR value format (where value is the next argument)
        elif arg.startswith("--") and arg[2:].isupper() and i + 1 < len(argv):
            var_name = arg[2:]
            value = argv[i + 1]
            # Only use this value if it doesn't look like another option
            if not value.startswith("--"):
                result[var_name] = value
                i += 1  # Skip the next argument since we used it as a value
        i += 1
    return result


cli_vars = find_all_caps_cli_vars()

########################################################
## Define defaults
########################################################
defaults = {}

# Absolute path to root directory of the project
if "BASE_DIR" in cli_vars:
    defaults["BASE_DIR"] = Path(cli_vars["BASE_DIR"])
else:
    defaults["BASE_DIR"] = Path(__file__).absolute().parent.parent


# OS type
def get_os():
    os_name = system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin" or os_name == "Linux":
        return "nix"
    else:
        return "unknown"


if "OS_TYPE" in cli_vars:
    defaults["OS_TYPE"] = cli_vars["OS_TYPE"]
else:
    defaults["OS_TYPE"] = get_os()


## Dates
defaults["BUFFER_START_DATE"] = datetime.strptime("1925-01-01", "%Y-%m-%d")
defaults["REPLICATION_START_DATE"] = datetime.strptime("1929-01-01", "%Y-%m-%d")
defaults["REPLICATION_END_DATE"] = datetime.strptime("2015-12-31", "%Y-%m-%d")
defaults["EXTENSION_START_DATE"] = datetime.strptime("2016-01-01", "%Y-%m-%d")
defaults["EXTENSION_END_DATE"] = datetime.strptime("2025-12-31", "%Y-%m-%d")


## File paths
def if_relative_make_abs(path):
    """
    If a relative path is given, make it absolute, assuming
    that it is relative to the project root directory (BASE_DIR)
    """
    path = Path(path)
    if path.is_absolute():
        abs_path = path.resolve()
    else:
        abs_path = (defaults["BASE_DIR"] / path).resolve()
    return abs_path


defaults = {
    "DATA_DIR": if_relative_make_abs(Path("_data")),
    "MANUAL_DATA_DIR": if_relative_make_abs(Path("data_manual")),
    "OUTPUT_DIR": if_relative_make_abs(Path("_output")),
    **defaults,
}


# Additional subdirectories for data and output
defaults["RAW_DATA_DIR"] = if_relative_make_abs(Path("_data/raw_data"))
defaults["PROCESSED_DATA_DIR"] = if_relative_make_abs(Path("_data/processed_data"))
defaults["DATA_DICTIONARY_DIR"] = if_relative_make_abs(Path("_data/data_dictionaries"))


def config(
    var_name,
    default=None,
    cast=None,
    settings_py_defaults=defaults,
    cli_vars=cli_vars,
    convert_dir_vars_to_abs_path=True,
):
    """
    Config defines a variable that can be used in the project. The definition of variables follows
    an order of precedence:
    1. Command line arguments
    2. Environment variables
    3. Settings.py file
    4. Defaults defined in-line in the local file
    5. Error
    """

    # 1. Command line arguments (highest priority)
    if var_name in cli_vars and cli_vars[var_name] is not None:
        value = cli_vars[var_name]
        # Apply cast if provided
        if cast is not None:
            value = cast(value)
        if "DIR" in var_name and convert_dir_vars_to_abs_path:
            value = if_relative_make_abs(Path(value))
        return value

    # 2. Environment variables through decouple
    # Use decouple but with a sentinel default to detect if it was found
    env_sentinel = object()
    env_value = _config(var_name, default=env_sentinel)
    if env_value is not env_sentinel:
        # Found in environment
        if cast is not None:
            env_value = cast(env_value)
        if "DIR" in var_name and convert_dir_vars_to_abs_path:
            env_value = if_relative_make_abs(Path(env_value))
        return env_value

    # 3. Settings.py defaults dictionary
    if var_name in defaults:
        default_value = defaults[var_name]
        # If default_value is directly usable (not a dict with metadata)
        if cast is not None:
            default_value = cast(default_value)
        return default_value

    # 4. Use the default value provided in the local file. Error if not found
    try:
        return _config(var_name, default=default, cast=cast)
    except Exception as e:
        raise ValueError(
            f"Configuration variable '{var_name}' is not defined. "
            f"Please set it via:\n"
            f"  1. Command line: --{var_name}=value\n"
            f"  2. Environment variable: export {var_name}=value\n"
            f"  3. .env file: {var_name}=value\n"
            f"Original error: {e}"
        ) from e


def create_directories():
    config("DATA_DIR").mkdir(parents=True, exist_ok=True)
    config("OUTPUT_DIR").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
