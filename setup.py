"""Create instructions to build the MRC package."""

import os
from setuptools import find_packages, setup
# import runpy

base_dir = os.path.dirname(os.path.abspath(__file__))
# MRCpy = runpy.run_path(os.path.join
#                                           (base_dir,
#                                            'MRCpy',
#                                            '__init__.py'))


# Read version without importing the module (to avoid dependency issues)
version = None
init_file = os.path.join(base_dir, 'MRCpy', '__init__.py')
with open(init_file, 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

if version is None:
    raise RuntimeError("Unable to find version string in MRCpy/__init__.py")


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
    # Make path relative to setup.py location
    filepath = os.path.join(base_dir, filename)
    with open(filepath) as input_file:
        lines = input_file.read().splitlines()
    
    # Filter out comments, empty lines, and lines starting with #
    requirements = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            # Remove inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:  # Check again after removing inline comments
                requirements.append(line)
    
    return requirements


if __name__ == '__main__':

    requirements = parse_requirements_file('requirements.txt')
    install_requires = []

    # Add all the requirements.
    for requirement in requirements:
        install_requires.append(requirement)

    # Define optional dependencies
    extras_require = {
        'lmrc': [
            'pycddlib>=3.0.2',
        ],
        'all': [
            'pycddlib>=3.0.2',
        ]
    }

    # Read README
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name="MRCpy",
        # version=MRCpy['__version__'],
        version=version,
        install_requires=install_requires,
        extras_require=extras_require,
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
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11"
        ],
        data_files=[
            "README.md"
        ],
        test_suite='tests',
        include_package_data=True,
        package_data={'': ['datasets/descr/*'],
                      'MRCpy': ['datasets/data/adult.csv',
                                'datasets/data/credit.csv',
                                'datasets/data/diabetes.csv',
                                'datasets/data/ecoli.csv',
                                'datasets/data/glass.csv',
                                'datasets/data/haberman.csv',
                                'datasets/data/indianLiverPatient.csv',
                                'datasets/data/iris.csv',
                                'datasets/data/letter-recognition.csv',
                                'datasets/data/mammographic.csv',
                                'datasets/data/optdigits.csv',
                                'datasets/data/redwine.csv',
                                'datasets/data/satellite.csv',
                                'datasets/data/segment.csv',
                                'datasets/data/usenet2.csv',
                                'datasets/data/comp-vs-sci_shortTrain.csv',
                                'datasets/data/comp-vs-sci_shortTest.csv',
                                ]},
        python_requires='>=3.9',
        zip_safe=False
    )
