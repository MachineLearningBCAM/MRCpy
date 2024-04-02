
"""Create instructions to build the MRC package."""

import os
import runpy
from setuptools import find_packages, setup

base_dir = os.path.dirname(os.path.abspath(__file__))
MRCpy = runpy.run_path(os.path.join
                                          (base_dir,
                                           'MRCpy',
                                           '__init__.py'))


def parse_requirements_file(filename):
    """
    Read the lines of the requirements file.

    Parameters
    ----------
    filename : str
        The filename to read

    Returns
    -------
        Array of lines of the requirements file.
    """
    with open(filename) as input_file:
        return input_file.read().splitlines()


if __name__ == '__main__':

    requirements = parse_requirements_file('requirements.txt')
    install_requires = []

    # Add all the requirements.
    for requirement in requirements:
        install_requires.append(requirement)

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name="MRCpy",
        version=MRCpy['__version__'],
        install_requires=install_requires,
        description="Minimax Risk Classification",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/MachineLearningBCAM/MRCpy",
        packages=find_packages(),
        # py_modules=["MRCpy", "MRCpy.phi"],
                    # "phi.PhiGaussian", "phi.PhiLinear", "phi.PhiThreshold"],
        # package_dir={'': 'minimax_risk_classifiers'},
        classifiers=[
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Machine Learning",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        data_files=[
            "README.md"
        ],
        test_suite='tests',
        include_package_data=True,
        package_data={'': ['datasets/data/*', 'datasets/descr/*']},
        python_requires='>=3.6',
        zip_safe=False
    )
