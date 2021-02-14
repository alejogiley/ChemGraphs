#! /usr/bin/env bash

python -m pip install -e .[develop]

x86='/usr/lib/x86_64-linux-gnu'
url='https://anaconda.org/rdkit/rdkit/2018.09.1.0/download/linux-64/rdkit-2018.09.1.0-py36h71b666b_1.tar.bz2'

# download & extract
curl -L $url | tar xj lib

# move to python packages directory
mv lib/python3.6/site-packages/rdkit /usr/local/lib/python3.6/dist-packages/
mv lib/*.so.* $x86/

# rdkit need libboost
ln -s $x86/libboost_python3-py36.so.1.65.1 $x86/libboost_python3.so.1.65.1
