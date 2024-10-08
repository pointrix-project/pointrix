from setuptools import find_packages, setup

setup(
    name='pointrix',
    version='1.0.0',    
    description='Pointrix: a differentiable point-based rendering libraries',
    url='https://github.com/pointrix-project/Pointrix',
    author='NJU-3dv',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ]
)