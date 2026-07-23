from setuptools import setup, find_packages

setup(
    name='MOTorNOT',
    version='1.0.0',
    description='Semiclassical MOT/dipole-trap simulation library with a CPU/GPU (CuPy) backend',
    author='Robert Fasano',
    author_email='robert.j.fasano@colorado.edu',
    packages=find_packages(),
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'plotly',
        'attrs',
        'pyyaml',
    ],
    extras_require={
        # Optional GPU acceleration (NVIDIA CUDA 12.x). The [ctk] extra of CuPy
        # provides the CUDA headers needed to compile kernels.
        'gpu': ['cupy-cuda12x[ctk]'],
    },
    include_package_data=True,
    package_data={'MOTorNOT': ['parameters.yml']},
)
