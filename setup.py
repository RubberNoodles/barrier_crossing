from setuptools import setup

setup(
  name = "barrier_crossing",
  version = "0.0.2",
  url = "https://github.com/RubberNoodles/barrier_crossing.git",
  packages = setuptools.find_packages(),
  install_requires = [
    "numpy",
    "matplotlib",
    "scipy",
    "tensorflow_probability",
    "tqdm",
    "typing",
    "pandas",
    "jaxlib", 
    "jax",
    "jraph",
    "jax-md"
  ]
)
