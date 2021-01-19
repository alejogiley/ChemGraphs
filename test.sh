#! /usr/bin/env bash

set -ex

if [[ -z $VIRTUAL_ENV ]]; then
	source /Applications/Anaconda/anaconda3/etc/profile.d/conda.sh
	conda activate env
fi

python -m pip install -e .[develop]
python setup.py nosetests --no-skip || error=1

if [[ $error -ne 1 ]]; then
	python setup.py flake8
fi

exit $error
