from pathlib import Path

from setuptools import setup, find_packages

_HERE = Path(__file__).resolve().parent
_LONG_DESCRIPTION = (_HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="pys7k",
    version="0.1.0",
    license="BSD-3-Clause",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
    ],
    description=("Pure-Python library and command line utility for indexing, reading, and writing "
                 "Teledyne RESON s7k sonar data files."),
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/valschmidt/pys7k",
    project_urls={
        "Source": "https://github.com/valschmidt/pys7k",
        "Issue Tracker": "https://github.com/valschmidt/pys7k/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "s7k.py=PYS7K.s7k:main",
        ],
    },
    keywords="hydrography multibeam sonar teledyne reson s7k 7k",
    author="Val Schmidt",
    author_email="Val.Schmidt@unh.edu",
)
