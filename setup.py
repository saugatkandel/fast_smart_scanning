from setuptools import setup
from setuptools import find_packages

setup(
    name='sladsnet',
    version='0.0.1',
    author='-',
    author_email='-',
    packages=find_packages(),
    scripts=[],
    description='SLADS-Net',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "joblib"
    ],
)