import os

import setuptools


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r", encoding="utf-8") as fh:
        return fh.read()


setuptools.setup(
    name="embedin",
    version="0.1.1",
    author="EmbedInAI",
    author_email="EmbedInAI@gmail.com",
    description="A lightweight vector database",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/EmbedInAI/EmbedInDB",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=read("requirements.txt"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
