from __future__ import absolute_import

import logging
import pkgutil

import pkg_resources

# Declare namespace package both ways for setuptools
__path__ = pkgutil.extend_path(__path__, __name__)
pkg_resources.declare_namespace(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())
