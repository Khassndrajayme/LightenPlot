"""
Setup configuration for LightenPlot package.
Fixed version - no metadata errors.
"""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "A lightweight Python library for easy data visualization"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0"
    ]

setup(
    name="LightenPlot",
    version="0.3.0",
    author="Group 5 - RichieClan",
    author_email="khassandrajayme@gmail.com",
    description="A lightweight Python library for easy data visualization with OOP principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khassndrajayme/lightenplot",
    project_urls={
        "Bug Tracker": "https://github.com/khassndrajayme/lightenplot/issues",
        "Documentation": "https://github.com/khassndrajayme/lightenplot#readme",
        "Source Code": "https://github.com/khassndrajayme/lightenplot",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="visualization plotting data-science matplotlib seaborn oop",
    license_files = "LICENSE",
    include_package_data=True,
    zip_safe=False,
)