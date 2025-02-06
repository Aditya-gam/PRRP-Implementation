from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="prrp_implementation",
    version="1.0.0",
    author="Aditya Gambhir",
    author_email="adityajune196@gmail.com",
    description="Implementation of the P-Regionalization through Recursive Partitioning (PRRP) algorithm for spatial and graph-based regionalization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aditya-gam/PRRP-Implementation.git",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Graph Theory",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run_spatial_prrp=spatial_prrp:main",
            "run_graph_prrp=graph_prrp:main",
        ],
    },
    zip_safe=False,
)
