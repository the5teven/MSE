from setuptools import setup, find_packages

setup(
    name="sme",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Simulated Method of Embeddings (SME) Framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sme",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch>=1.5.0',
        'faiss-cpu',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)