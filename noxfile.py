# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


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

from os.path import abspath, dirname, join
from pathlib import Path

import nox

BASE_DIR = Path(__file__).parent
SOURCE_FILES = ("setup.py", "noxfile.py", "opensearch_py_ml/", "utils/", "tests/")

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
    "opensearch_py_ml/ml_commons_integration/__init__.py",
    "opensearch_py_ml/ml_commons_integration/ml_common_client.py",
    "opensearch_py_ml/ml_commons_integration/ml_common_utils.py",
    "opensearch_py_ml/ml_commons_integration/load/__init__.py",
    "opensearch_py_ml/ml_commons_integration/load/ml_common_load_client.py",
    "opensearch_py_ml/ml_commons_integration/predict/__init__.py",
    "opensearch_py_ml/ml_commons_integration/predict/ml_common_predict_client.py",
    "opensearch_py_ml/plotting/_matplotlib/__init__.py",
)


@nox.session(reuse_venv=True)
def format(session):
    session.install("black", "isort", "flynt")
    session.run("python", "utils/lint/license-headers.py", "fix", *SOURCE_FILES)
    session.run("flynt", *SOURCE_FILES)
    session.run("black", "--target-version=py38", *SOURCE_FILES)
    session.run("isort", "--profile=black", *SOURCE_FILES)
    lint(session)


@nox.session(reuse_venv=True)
def lint(session):
    # Install numpy to use its mypy plugin
    # https://numpy.org/devdocs/reference/typing.html#mypy-plugin
    session.install("black", "flake8", "mypy", "isort", "numpy")
    session.run("python", "utils/lint/license-headers.py", "check", *SOURCE_FILES)
    session.run("black", "--check", "--target-version=py38", *SOURCE_FILES)
    session.run("isort", "--check", "--profile=black", *SOURCE_FILES)
    session.run("flake8", "--ignore=E501,W503,E402,E712,E203", *SOURCE_FILES)

    # TODO: When all files are typed we can change this to .run("mypy", "--strict", "opensearch_py_ml/")
    # TODO: Commenting this for now. As in windows subprocess.run don't find the process for mypy. We will revisit later
    # session.log("mypy --show-error-codes --strict opensearch_py_ml/")
    # for typed_file in TYPED_FILES:
    #     if not os.path.isfile(typed_file):
    #         session.error(f"The file {typed_file!r} couldn't be found")
    #     process = subprocess.run(
    #         ["mypy", "--show-error-codes", "--strict", typed_file],
    #         env=session.env,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT,
    #     )
    #     # Ensure that mypy itself ran successfully
    #     assert process.returncode in (0, 1)
    #
    #     errors = []
    #     for line in process.stdout.decode().split("\n"):
    #         filepath = line.partition(":")[0]
    #         if filepath in TYPED_FILES:
    #             errors.append(line)
    #     if errors:
    #         session.error("\n" + "\n".join(sorted(set(errors))))


@nox.session(python=["3.8", "3.9", "3.10"])
@nox.parametrize("pandas_version", ["1.5.2"])
def test(session, pandas_version: str):
    session.install(
        "-r",
        "requirements-dev.txt",
        "--timeout",
        "1500",
    )
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")
    session.run("python", "-m", "setup_tests")

    junit_xml = join(abspath(dirname(__file__)), "junit", "opensearch-py-ml-junit.xml")
    codecov_xml = join(
        abspath(dirname(__file__)), "junit", "opensearch-py-ml-codecov.xml"
    )

    pytest_args = (
        "python",
        "-m",
        "pytest",
        "--cov-report=term-missing",
        "--cov",
        "--cov-config=setup.cfg",
        "--doctest-modules",
        "--nbval",
        f"--junitxml={junit_xml}",
        f"--cov-report=xml:{codecov_xml}",
    )

    session.run(
        *pytest_args,
        *(session.posargs or ("opensearch_py_ml/", "tests/")),
    )


@nox.session(python=["3.9"])
@nox.parametrize("pandas_version", ["1.5.2"])
def docs(session, pandas_version: str):
    # Run this so users get an error if they don't have Pandoc installed.
    session.install("-r", "docs/requirements-docs.txt")
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")

    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)


# While nox is typically used for automating testing, in this case, we utilize it
# to automate the action workflow, leveraging its ability to set up the environment
# required for model autotracing.
@nox.session(python=["3.9"])
@nox.parametrize("pandas_version", ["1.5.2"])
def trace(session, pandas_version: str):
    session.install(
        "-r",
        "requirements-dev.txt",
        "--timeout",
        "1500",
    )
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")

    session.run(
        "python",
        "utils/model_uploader/model_autotracing.py",
        *(session.posargs),
    )
