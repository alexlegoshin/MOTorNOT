from setuptools import setup, find_packages

setup(
    name='MOTorNOT',
    version='0.1',
    description='Magneto-optical trap simulation package',
    author='Robert Fasano',
    author_email='robert.j.fasano@colorado.edu',
    packages=find_packages(),
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'plotly',
        'attrs',
        'pyyaml',
    ],
    include_package_data=True,
    package_data={'MOTorNOT': ['parameters.yml']},
)
