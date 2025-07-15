from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = []

setup(
    name="tdmpc2",
    version="0.0",
    author="Nicklas Hansen",
    url="https://github.com/nicklashansen/tdmpc2",
    description="TD-MPC2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.7",
    install_requires=core_requirements,
)
