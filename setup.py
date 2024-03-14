from setuptools import setup, find_packages

setup(
    name='LDHPbuilder',
    version='0.1',
    packages=find_packages(),
    description='random generation of 2D HOIPs',
    author='Will J Baldwin',
    author_email='wjb48@cam.ac.uk',
    license='MIT',
    url='https://github.com/WillBaldwin0/LDHP-builder',
    install_requires=[
        "numpy", "scipy", "ase",
    ]
)