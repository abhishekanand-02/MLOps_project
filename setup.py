from setuptools import setup, find_packages

setup(
    name="gemstone_price_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "flask"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "tox"
        ]
    },
    entry_points={
        "console_scripts": [
            "gemstone_price_prediction=src.pipelines.training_pipeline:main"
        ]
    }
)
