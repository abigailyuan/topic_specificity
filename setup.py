from setuptools import setup, find_packages

setup(
    name="topic_specificity",
    version="0.1.0",
    description="A Python package for calculating topic specificity for LDA, HDP, and LSA models.",
    author="Meng Yuan",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn"
        # add 'gensim' if you're using gensim models
    ],
    python_requires='>=3.6',
)
