#!/bin/bash
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# Define your environment name and requirements file
# Default values
ENV_NAME="epiclass_env"
REQUIREMENTS_FILE="requirements/minimal_requirements.txt"
LOCAL_MODULE_PATH="src/python"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--env) ENV_NAME="$2"; shift ;;
        -r|--req) REQUIREMENTS_FILE="$2"; shift ;;
        -s|--source) LOCAL_MODULE_PATH="$2"; shift ;;
        -h|--help) echo "Usage: $0 [-e virtual_env_name] [-r requirements_file] [-l local_module_path]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1. Use -h or --help for usage"; exit 1 ;;
    esac
    shift
done

# Check if Python is installed
if ! command -v python3 > /dev/null 2>&1; then
    echo "Python3 is not installed. Please install Python3 and rerun the script."
    exit 1
fi

# Check if venv is installed
if ! python3 -c "import venv" > /dev/null 2>&1; then
    echo "Python3 venv is not installed. Please install Python3 venv and rerun the script."
    exit 1
fi

# Check if pip is installed
if ! command -v pip > /dev/null 2>&1; then
    echo "pip is not installed. Please install pip and rerun the script."
    exit 1
fi

# Create a Python virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Check if requirements file exists
if [ ! -f $REQUIREMENTS_FILE ]; then
    echo "requirements.txt does not exist. Please check the path."
    exit 1
fi

# Install packages from requirements.txt
if ! pip install -r $REQUIREMENTS_FILE; then
    echo "There was an error installing the packages from the requirements.txt file."
    exit 1
fi

# Check if local module path exists
if [ ! -d $LOCAL_MODULE_PATH ]; then
    echo "The specified local module path does not exist. Please check the path."
    exit 1
fi

# Install local module as editable
if ! pip install -e $LOCAL_MODULE_PATH; then
    echo "There was an error installing the local module."
    exit 1
fi

echo "Setup Complete"
