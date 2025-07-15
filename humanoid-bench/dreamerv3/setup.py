import pathlib

import setuptools

extras = {}
extras["tests"] = ["pytest", "parameterized", "cloudpickle"]
extras["optional"] = ["rich", "pillow", "tensorflow", "cloudpickle"]
extras["envs"] = [
    # "gym==0.19.0",
    "atari_py",
    "crafter",
    "dm_control",
    "robodesk",
    "procgen",
    "bsuite",
]
extras["dreamerv2"] = ["tensorflow", "tensorflow_probability", "ruamel.yaml"]
extras["all"] = sorted(set(sum([v for v in extras.values()], [])))

setuptools.setup(
    name="embodied",
    version="0.0.0",
    author="Danijar Hafner",
    author_email="mail@danijar.com",
    description="Fast reinforcement learning research",
    url="http://github.com/danijar/embodied",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy"],
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
