from setuptools import setup, find_packages

setup(
    name="FNN-1",  # This is what pip will use - with dash
    packages=["FNN_1"],  # This points to your actual package directory - with underscore
    package_data={"": ["*.py"]},
    version="0.1",
    description="FNN implementation for sentiment analysis and clustering",
    install_requires=[
        "torch",
        "keras",
        "tensorflow",
        "numpy",
        "scikit-learn",
        "transformers"
    ],
)