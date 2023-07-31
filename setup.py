#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []

test_requirements = ['pytest>=3', ]

setup(
    author="Christain Loving[D[D[D[D[D[D[Dia[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C",
    author_email='christian70401@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="This package makes your wildest dreams come true",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ras2d_viz',
    name='ras2d_viz',
    packages=find_packages(include=['ras2d_viz', 'ras2d_viz.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Clovin4/ras2d_viz',
    version='0.1.0',
    zip_safe=False,
)