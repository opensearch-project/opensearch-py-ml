#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import os
import subprocess
from pathlib import Path

import nox

BASE_DIR = Path(__file__).parent
SOURCE_FILES = ("setup.py", "noxfile.py", "opensearch_py_ml/", "docs/", "utils/", "tests/", "bin/")

# Whenever type-hints are completed on a file it should
# be added here so that this file will continue to be checked
# by mypy. Errors from other files are ignored.
TYPED_FILES = (
    "opensearch_py_ml/actions.py",
    "opensearch_py_ml/arithmetics.py",
    "opensearch_py_ml/common.py",
    "opensearch_py_ml/etl.py",
    "opensearch_py_ml/filter.py",
    "opensearch_py_ml/index.py",
    "opensearch_py_ml/query.py",
    "opensearch_py_ml/tasks.py",
    "opensearch_py_ml/utils.py",
    "opensearch_py_ml/groupby.py",
    "opensearch_py_ml/operations.py",
    "opensearch_py_ml/ndframe.py",
    "opensearch_py_ml/ml/__init__.py",
    "opensearch_py_ml/ml/_optional.py",
    "opensearch_py_ml/ml/_model_serializer.py",
    "opensearch_py_ml/ml/ml_model.py",
    "opensearch_py_ml/ml/pytorch/__init__.py",
    "opensearch_py_ml/ml/pytorch/_pytorch_model.py",
    "opensearch_py_ml/ml/pytorch/transformers.py",
    "opensearch_py_ml/ml/transformers/__init__.py",
    "opensearch_py_ml/ml/transformers/base.py",
    "opensearch_py_ml/ml/transformers/lightgbm.py",
    "opensearch_py_ml/ml/transformers/sklearn.py",
    "opensearch_py_ml/ml/transformers/xgboost.py",
    "opensearch_py_ml/plotting/_matplotlib/__init__.py",
)


@nox.session(reuse_venv=True)
def format(session):
    session.install("black", "isort", "flynt")
    session.run("python", "utils/license-headers.py", "fix", *SOURCE_FILES)
    session.run("flynt", *SOURCE_FILES)
    session.run("black", "--target-version=py37", *SOURCE_FILES)
    session.run("isort", "--profile=black", *SOURCE_FILES)
    lint(session)


@nox.session(reuse_venv=True)
def lint(session):
    # Install numpy to use its mypy plugin
    # https://numpy.org/devdocs/reference/typing.html#mypy-plugin
    session.install("black", "flake8", "mypy", "isort", "numpy")
    session.install("--pre", "elasticsearch>=8.0.0a1,<9")
    session.run("python", "utils/license-headers.py", "check", *SOURCE_FILES)
    session.run("black", "--check", "--target-version=py37", *SOURCE_FILES)
    session.run("isort", "--check", "--profile=black", *SOURCE_FILES)
    session.run("flake8", "--ignore=E501,W503,E402,E712,E203", *SOURCE_FILES)

    # TODO: When all files are typed we can change this to .run("mypy", "--strict", "opensearch_py_ml/")
    session.log("mypy --show-error-codes --strict opensearch_py_ml/")
    for typed_file in TYPED_FILES:
        if not os.path.isfile(typed_file):
            session.error(f"The file {typed_file!r} couldn't be found")
        process = subprocess.run(
            ["mypy", "--show-error-codes", "--strict", typed_file],
            env=session.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Ensure that mypy itself ran successfully
        assert process.returncode in (0, 1)

        errors = []
        for line in process.stdout.decode().split("\n"):
            filepath = line.partition(":")[0]
            if filepath in TYPED_FILES:
                errors.append(line)
        if errors:
            session.error("\n" + "\n".join(sorted(set(errors))))


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
@nox.parametrize("pandas_version", ["1.2.0", "1.3.0"])
def test(session, pandas_version: str):
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")
    session.run("python", "-m", "tests.setup_tests")

    pytest_args = (
        "python",
        "-m",
        "pytest",
        "--cov-report=term-missing",
        "--cov=opensearch_py_ml/",
        "--cov-config=setup.cfg",
        "--doctest-modules",
        "--nbval",
    )

    # PyTorch doesn't support Python 3.10 yet
    if session.python == "3.10":
        pytest_args += ("--ignore=opensearch_py_ml/ml/pytorch",)
    session.run(
        *pytest_args,
        *(session.posargs or ("opensearch_py_ml/", "tests/")),
    )

    # Only run during default test execution
    if not session.posargs:
        session.run(
            "python",
            "-m",
            "pip",
            "uninstall",
            "--yes",
            "scikit-learn",
            "xgboost",
            "lightgbm",
        )
        session.run("pytest", "tests/ml/")


@nox.session(reuse_venv=True)
def docs(session):
    # Run this so users get an error if they don't have Pandoc installed.
    session.run("pandoc", "--version", external=True)

    session.install("-r", "docs/requirements-docs.txt")
    session.install(".")

    # See if we have an Elasticsearch cluster active
    # to rebuild the Jupyter notebooks with.
    es_active = False
    try:
        from elasticsearch import ConnectionError, Elasticsearch

        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            if not es.indices.exists(index="flights"):
                session.run("python", "-m", "tests.setup_tests")
            es_active = True
        except ConnectionError:
            pass
    except ImportError:
        pass

    # Rebuild all the example notebooks inplace
    if es_active:
        session.install("jupyter-client", "ipykernel")
        for filename in os.listdir(BASE_DIR / "docs/sphinx/examples"):
            if (
                filename.endswith(".ipynb")
                and filename != "introduction_to_eland_webinar.ipynb"
            ):
                session.run(
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--inplace",
                    "--execute",
                    str(BASE_DIR / "docs/sphinx/examples" / filename),
                )

    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
