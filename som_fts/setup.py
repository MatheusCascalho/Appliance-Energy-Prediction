from setuptools import setup
import setuptools


# read description
with open("README.md", "r") as fh:
    long_description = fh.read()

# read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="som_fts",
    version="1.0",
    author="MATHEUS CASCALHO",
    author_email="cascalhom@gmail.com",
    long_description=long_description,
    install_requires=required,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['som_fts']),
    include_package_data=True,
    description="SOM FTS TOOLS"
)