from setuptools import setup, find_packages
import os

# Read the contents of your README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="topic-specificity",
    version="0.1.2",
    author="Meng Yuan",
    author_email="meng.yuan@unimelb.edu.au",
    maintainer="Meng Yuan",
    maintainer_email="meng.yuan@unimelb.edu.au",
    description="Calculate topic specificity for LDA, LSA & HDP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abigailyuan/topic_specificity",
    project_urls={
        "Paper": "https://doi.org/10.1145/3578337.3605118",
        "CITATION": "https://github.com/abigailyuan/topic_specificity/blob/main/CITATION.cff"
    },
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
