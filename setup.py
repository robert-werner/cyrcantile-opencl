from setuptools import setup, find_packages

setup(
    name="cyrcantile",
    version="0.2.0",
    description="OpenCL-accelerated spherical-mercator tile utilities",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"cyrcantile": ["_kernels.cl"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "opencl": ["pyopencl>=2021.1"],
        "vulkan": ["wgpu>=0.17"],
        "cpu-speed": ["numba>=0.60"],
        "cuda": ["numba>=0.60"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
