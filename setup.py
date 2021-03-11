from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="pgportfolio-pytorch",
    version="1.0.0",
    description="",
    long_description=readme,
    author="Muhan Li",
    author_email="",
    packages=find_packages(exclude=("tests", "docs"),
                           include=("matplotlib", "tensorflow", "tflearn", "pandas",
                                    "pandas", "cvxopt", "scipy")))