from setuptools import setup
import setuptools


# read description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Appliance-Energy-Prediction",
    version="1.0",
    author="MATHEUS CASCALHO",
    author_email="cascalhom@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Appliance Energy Prediction"
)