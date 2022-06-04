#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


requirements = []

test_requirements = []

setup(
    author="Arvid Kingl",
    author_email="akingl2016@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python Boilerplate contains all the boilerplate you need to\\ create a Python package.",
    install_requires=requirements,
    long_description="",
    include_package_data=True,
    keywords="yomi_bot",
    name="yomi_bot",
    packages=find_packages(include=["yomi_bot", "yomi_bot.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/hcelion/yomi_bot",
    version="0.1.0",
    zip_safe=False,
)
