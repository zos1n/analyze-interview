from setuptools import setup, find_packages

setup(
    name="analyze-interview",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.20.0",
        "rich>=12.0.0",
    ],
    entry_points={
        "console_scripts": [
            "analyze-interview=analyze_interview.cli:main",
        ],
    },
)
