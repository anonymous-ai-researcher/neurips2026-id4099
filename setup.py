from setuptools import setup, find_packages

setup(
    name="fuzzycell",
    version="0.1.0",
    description="Differentiable Cell Ontology Reasoning for Graded Cell Type Annotation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scanpy>=1.9.0",
        "transformers>=4.30.0",
        "pyyaml>=6.0",
    ],
)
