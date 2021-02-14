#! /usr/bin/env bash

set -ex

if [[ -z $VIRTUAL_ENV ]]; then
	python -m venv env
	source env/bin/activate
fi

. develop.sh

python setup.py nosetests --no-skip || error=1

if [[ $error -ne 1 ]]; then
	python setup.py flake8
fi

exit $error
