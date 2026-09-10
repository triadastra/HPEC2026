from setuptools import setup, find_packages

setup(
    name="hpec2026",
    version="0.1.0",
    description="Multidimensional long-series forecasting on a U.S. Census trade lattice",
    author="HPEC 2026 authors",
    python_requires=">=3.11,<3.12",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "transformers>=4.20.0",
        "einops>=0.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
        "cuda": [
            "triton>=3.5.0", "tilelang==0.1.8", "apache-tvm-ffi<=0.1.9",
            "quack-kernels>=0.3.4", "ninja>=1.11.0", "packaging>=24.0",
        ],
    },
)
