from setuptools import setup, find_packages

setup(
    name="elerpo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "integration": ["torch"],
    },
    python_requires=">=3.9",
)
