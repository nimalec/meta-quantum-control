"""
Setup script for meta_quantum_control package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meta_quantum_control",
    version="1.0.0",
    author="Nima Leclerc",
    author_email="nleclerc@mitre.org",
    description="Meta-RL for Quantum Control with Optimality Gap Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/meta-quantum-control",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "ipykernel>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-meta=experiments.train_meta:main",
            "train-robust=experiments.train_robust:main",
            "eval-gap=experiments.eval_gap:main",
        ],
    },
)
