#!/usr/bin/env python
"""Setup configuration for MathCore package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mathstack",
    version="3.2.0",
    author="MathCore Contributors",
    author_email="info@mathcore.dev",
    description="Advanced mathematics library with zero external dependencies - symbolic algebra, geometry, calculus, statistics, optimization, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mathcore/mathcore",
    project_urls={
        "Bug Tracker": "https://github.com/mathcore/mathcore/issues",
        "Documentation": "https://mathcore.readthedocs.io",
        "Source Code": "https://github.com/mathcore/mathcore",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
