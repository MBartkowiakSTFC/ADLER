"""
Setuptools setup script for ADLER

While it is expected that pip will be used install ADLER, it can also be
installed with the command::

    python setup.py install .

"""

import sys

from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

# Check for valid Python version
if sys.version_info[:2] < (3, 6):
    print('ADLER requires Python 3.6 or better. Python {0:d}.{1:d}'
          ' detected'.format(*sys.version_info[:2]))

packages_test=find_packages()
setup(
    name="ADLER-PEAXIS",
    version="4.1",
    description=('Data reduction software for the PEAXIS RIXS instrument.'),
    packages=find_packages(),
    author="Maciej Bartkowiak",
    # author_email="support@mdmcproject.org",
    # url="https://mdmcproject.org/",
    # download_url="https://github.com/MDMCproject",
    python_requires='>=3.6',
    install_requires=[pr.requirement for pr in parse_requirements('requirements.txt', session= PipSession())],
    # entry_points={"console_scripts": ['MDMC = MDMC.utilities.cli:main']},
    include_package_data=True
)
