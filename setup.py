from setuptools import setup, find_packages

setup(
    name="barrier_crossing",
    version="0.1.0",
    packages=["barrier_crossing"],
    install_requires=[],  # Add dependencies if needed
    author="Oliver Cheng, Zosia Adamska, Megan Engel",
    description="Iteratively Reconstructing Using Nonequilibrium Pulling.",
    url="https://github.com/rubbernoodles/barrier_crossing",  # Update with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)