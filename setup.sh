#!/bin/bash

# Before running this script, install miniconda or anaconda with Python 3
# https://conda.io/docs/user-guide/install/macos.html

# After installing either miniconda or anaconda, open a new terminal code so conda is on the path
# Setting the current directory into this directory seems to help this script successfully

# After this script runs, you may have to start a new terminal window

# Setup the environment for the classroom. TeachOps FTW!
set -e # Exit immediately if a command exits with a non-zero status
set -x # Exit immediately if a pipeline exits with a non-zero status

# Environment name is from environment.yml
envname=$(sed '1!d' environment.yml | sed 's/^.* //')

# Create environment based on dependencies in environment.yml
conda update --name base conda -y 
conda env create --name $envname --force 

# Start environment
conda activate $envname

# Update during development
# conda update --all --yes 
# # Pin right before teaching
# conda env export > environment.yml

# Enable extensions
jupyter nbextension enable spellchecker/main
jupyter nbextension enable codefolding/main
jupyter nbextension install rise --py --sys-prefix
jupyter nbextension enable rise --py --sys-prefix
jupyter nbextension install --py jupyter_tabnine --user
jupyter nbextension enable --py jupyter_tabnine --user
jupyter serverextension enable --py jupyter_tabnine --user

# Install graphviz for visualizing trees on Mac using homebrew
brew install graphviz
# For Linux, download here https://graphviz.gitlab.io/download/



# Note: This shell script comes with no warranty or support. It is just an aid to help you create a reproducible enirovnment for the course. Your computer might not play nice with it, and this script depends on The Internet (which changes).

# Warnings can be ingored.

# If this script fails to run completely, copy n' paste each line into the terminal that is running the current environment.

# If get `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`
# Follow the directions
