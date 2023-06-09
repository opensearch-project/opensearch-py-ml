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

# flake8: noqa

from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))
about = {}
with open(path.join(here, "opensearch_py_ml", "_version.py"), "r", "utf-8") as f:
    exec(f.read(), about)

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: Apache Software License",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]

# Remove all raw HTML from README for long description
with open(path.join(here, "README.md"), "r", "utf-8") as f:
    lines = f.read().split("\n")
    last_html_index = 0
    for i, line in enumerate(lines):
        if line == "</p>":
            last_html_index = i + 1
    long_description = "\n".join(lines[last_html_index:])

extras = {}
extras["all"] = list({dep for deps in extras.values() for dep in deps})

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    maintainer=about["__maintainer__"],
    maintainer_email=about["__maintainer_email__"],
    license="Apache-2.0",
    classifiers=CLASSIFIERS,
    keywords="Opensearch opensearch_py_ml pandas python",
    packages=find_packages(include=["opensearch_py_ml", "opensearch_py_ml.*"]),
    project_urls={
        "Source Code": "https://github.com/opensearch-project/opensearch-py-ml",
        "Issue Tracker": "https://github.com/opensearch-project/opensearch-py-ml/issues",
    },
    install_requires=[
        "opensearch-py>=2",
        "pandas>=1.5,<3",
        "matplotlib>=3.6.0,<4",
        "numpy>=1.24.0,<2",
    ],
    python_requires=">=3.8",
    package_data={"opensearch_py_ml": ["py.typed"]},
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
