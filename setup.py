import codecs
import os
import re

from setuptools import setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "gcnn", "__init__.py")


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))


setup(
    name="gcnn",
    version=get_version(),
    description="Graph Neural Network model for binding affinity prediction",
    author="Alejandro Gil-Ley",
    url="https://github.com/alejogiley/ChemGraphs",
    scripts=["gcnn/apps/setup_dataset.py", "gcnn/apps/train_gcnn.py"],
)
