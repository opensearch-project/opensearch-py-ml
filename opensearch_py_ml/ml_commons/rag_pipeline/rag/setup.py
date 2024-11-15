from setuptools import setup, find_packages, find_namespace_packages




setup(
    name="rag_pipeline",
    version="0.1.0",
    packages=find_namespace_packages(include=['opensearch_py_ml', 'opensearch_py_ml.*']),
    entry_points={
        'console_scripts': [
        'rag=opensearch_py_ml.ml_commons.rag_pipeline.rag:main',
        ],
    },
)