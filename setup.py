from setuptools import setup
import setuptools


# read description
with open("README.md", "r") as fh:
    long_description = fh.read()

# read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="Appliance_Energy_Prediction",
    version="1.0",
    author="MATHEUS CASCALHO",
    author_email="cascalhom@gmail.com",
    long_description=long_description,
    install_requires=required,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['Appliance_Energy_Prediction']),
    include_package_data=True,
    description="Appliance Energy Prediction"
)