from setuptools import setup

setup(
  name = "barrier_crossing",
  version = "0.0.2",
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
  ],
  packages = setuptools.find_packages()
)
