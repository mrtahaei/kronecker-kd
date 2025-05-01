from setuptools import setup, find_packages

setup(
    name="kronecker-kd",
    version="0.1.0",
    description="Kronecker Knowledge Distillation",
    author="Marzieh",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "trl>=0.4.7",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "accelerate>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
) 