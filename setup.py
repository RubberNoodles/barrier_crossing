from setuptools import setup, find_packages

setup(
    name="barrier_crossing",
    version="0.1.0",
    packages=["barrier_crossing"],
    install_requires=[ "absl-py==1.1.0",
    "jax==0.4.16",
    "jax-md==0.2.8",
    "matplotlib==3.5.2",
    "numpy==1.22.4",
    "optax==0.1.7",
    "pandas==1.4.3",
    "scipy==1.7.3",
    "seaborn==0.13.2",
    "setuptools==65.6.3",  # Keeping the latest version
    "tensorflow-probability==0.22.0",
    "tqdm==4.64.1" 
    ],
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