"""Setup configuration for LightenPlot package."""

from setuptools import setup, find_packages
import pathlib

# Read README for long description
HERE = pathlib.Path(__file__).parent
# FIX: Added encoding='utf-8' to prevent UnicodeDecodeError on Windows
README = (HERE / "README.md").read_text(encoding='utf-8') if (HERE / "README.md").exists() else "LightenPlot - Simplified Data Visualization"

# Read requirements
with open('requirements.txt', encoding='utf-8') as f: # Added encoding='utf-8' for robustness
    requirements = f.read().splitlines()

setup(
    name='LightenPlot',
    version='0.1.0',
    author='Richieclan',
    author_email='khassandrajayme@gmail.com',
    description='A Python library that dramatically simplifies data visualization syntax',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/khassndrajayme/lightenplot',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    keywords='visualization plotting data-science matplotlib seaborn',
    project_urls={
        'Bug Reports': 'https://github.com/khassndrajayme/lightenplot/issues',
        'Source': 'https://github.com/khassndrajayme/lightenplot',
        'Documentation': 'https://github.com/khassndrajayme/lightenplot/blob/main/README.md',
    },
)

