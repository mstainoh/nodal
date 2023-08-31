from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="well_nodal_tools",  
    license='MIT',
    version="2.0.0",  
    description="nodal analysis tools for oil, gas and water wells",  
    url="https://github.com/mstainoh/nodal",
    author="Marcelo Stainoh", 
    author_email="mstainoh@gmail.com",
    classifiers=[ 
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Engineers",
        "Topic :: Petroleumn Engineering :: Calculation Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    project_urls={
        "Source": "https://github.com/mstainoh/nodal",
    },
)
