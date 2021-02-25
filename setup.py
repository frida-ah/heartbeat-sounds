from setuptools import setup, find_packages

setup(
    name="heartbeat-sounds",
    packages=find_packages(),
    version="0.1.0",
    description="This project contains the models for Mass Market Promotions CE forecasts.",
    author="PDF team",
    setup_requires=["wheel"],
    install_requires=[
        "pandas==1.0.1",
        "librosa==0.8",
        "numpy==1.20.0",
        "matplotlib==3.3.4",
    ],
    extras_require={
        "dev": [
            "pyspark==3.0.1",
            "pre-commit==2.2.0",
            "pytest-cov==2.10.0",
            "flake8==3.8.4",
            "black==20.8b1",
            "pytest==5.4.3",
        ]
    },
    python_requires=">=3.6",
)
