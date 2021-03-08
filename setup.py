from setuptools import setup, find_packages
import os


with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ScaledYOLOv4',
    version=os.getenv('0.1.0'),
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=required,
    url='https://gitlab.hq.braincreators.com/research/yolo',
    author='Ioannis Gatopoulos',
    author_email='johngatop@gmail.com',
    description='ScaledYOLOv4: Object Detection Algorithm.',
    long_description=long_description
)
