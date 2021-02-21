#! /usr/bin/env bash

set -ex

NAME="chemgraph"

if [[ -z $CONDA_DEFAULT_ENV ]]; then
	
	CONDA_BASE=$(conda info --base)
	ENVS=$(conda env list 2>&1)

	export PATH="$CONDA_BASE/bin:$PATH"
	source $CONDA_BASE/etc/profile.d/conda.sh

	if [[ $ENVS == *$NAME* ]]; then
		source activate $NAME
	else
		conda env create -f environment.yml
		source activate $NAME
	fi

else

	source activate $NAME
	if [[ $? -eq 0 ]]; then
        :
    else
    	conda env create -f environment.yml
    	source activate $NAME
    fi

fi

python -m pip install -e .[develop]

python setup.py nosetests --no-skip || error=1

if [[ $error -ne 1 ]]; then
	python setup.py flake8
fi

exit $error
