from setuptools import setup

setup(
    name="rnabang",
    packages=[
        'data',
        'model',
        'experiments'
    ],
    package_dir={
        'data': './data',
        'model': './model',
        'experiments': './experiments'
    },
)