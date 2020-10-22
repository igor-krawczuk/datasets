from setuptools import setup,find_packages

install_reqs=[
    "torch",
    "filelock",
    "numpy",
    "pillow",
]

setup(
    name='daneml',
    version='0.0.1',
    packages=find_packages("."),
    url='',
    install_requires=install_reqs,
    license='AGPLv3',
    author='Igor Krawczuk',
    author_email='igor.krawczuk@epfl.ch',
    description=' ML datasets I use in my research '
)
