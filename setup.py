#!/usr/bin/env python
import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('acceptability', '__init__.py')
long_description = read('README.md')

requirements = [
    'numpy',
    'pillow >= 4.1.1',
    'six',
    'torch',
    'torchtext'
]

setup_info = dict(
    # Metadata
    name='acceptability',
    version=VERSION,
    author='Amanpreet Singh and Alex Warstadt',
    author_email='apsdehal@gmail.com',
    url='https://github.com/nyu-mll/acceptability-judgments',
    description='Models for Grammaticality Judgments data',
    long_description=long_description,
    license='BSD',
    install_requires=requirements,

    # Package info
    packages=find_packages(),
    zip_safe=True,
)

setup(**setup_info)
