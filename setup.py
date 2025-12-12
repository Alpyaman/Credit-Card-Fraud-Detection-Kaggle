"""Setup configuration for fraud detection project."""

from setuptools import setup, find_packages

setup(
    name="fraud-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "pydantic",
        "python-multipart",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "httpx",
            "flake8",
        ]
    },
)
