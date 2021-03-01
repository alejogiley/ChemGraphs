#! /usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL='3'

set -ex

NAME="chemgraph"
SRC_DIR=${SRC_DIR:-$(pwd)}
CONDA_BASE=$(conda info --base)
ENVS=$(conda info --envs 2>&1)

# Activate Conda manager
if [[ -z $CONDA_DEFAULT_ENV ]]; then
	export PATH="$CONDA_BASE/bin:$PATH"
	source "$CONDA_BASE/etc/profile.d/conda.sh"

elif [[ $CONDA_DEFAULT_ENV != *$NAME* ]]; then
	# Check if Chemgraph enviroment is available
	# otherwise create enviroment
	if [[ $ENVS == *$NAME* ]]; then
		echo "Activating $NAME enviroment"
		source activate $NAME
	else
		echo "Creating $NAME enviroment"
		conda env create -f environment.yml
		source activate $NAME
	fi
fi

python -m pip install -e .[develop]
python setup.py nosetests --no-skip || error=1

if [[ $error -ne 1 ]]; then
	echo "Checking code style with black..."
	python -m black --line-length 100 --check "${SRC_DIR}"
	echo "Success!"

	echo "Type checking with mypy..."
	mypy --ignore-missing-imports gcnn
	echo "Success!"

	# echo "Checking code style with pylint..."
	# python -m pylint "${SRC_DIR}"/gcnn/ "${SRC_DIR}"/test/*.py
	# echo "Success!"
fi

exit $error
