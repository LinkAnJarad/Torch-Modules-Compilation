from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = "Compilation of Torch Modules from various ML papers"

setup(
    name="torch_modules_compilation",
    version=VERSION,
    author="Link An Jarad",
    description=DESCRIPTION,
    long_description_content_type = 'text/markdown',
    long_description = open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.7',
    ],
)
