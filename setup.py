from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = "Compilation of Torch Modules from various ML papers"

setup(
    name="torch_modules_compilation",
    version=VERSION,
    author="Link An Jarad",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch']
)
