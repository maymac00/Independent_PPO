from setuptools import setup, find_packages

setup(name='IPPO',
      version='0.1',
      install_requires=['gym', 'matplotlib', 'numpy', 'torch'],
      packages=find_packages(),
      )