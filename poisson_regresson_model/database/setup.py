from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Experiment Database'
LONG_DESCRIPTION = 'Database for assiging a unique ID to each experiment ' + \
                   'run mouse ID for mice'
setup(
    name='database',
    version=VERSION,
    author='jack',
    author_email='jack.bowler@utah.edu',
    packages=['database'],
    install_requires=['pyyaml'],
    keywords=[],
    classifiers=[],
    package_data={'':['config.yaml']}
)
