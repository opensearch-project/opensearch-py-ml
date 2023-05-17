<p center><img src="https://github.com/opensearch-project/opensearch-py/raw/main/OpenSearch.svg" height="64px"/></p>
<h1 center>OpenSearch-py-ml Developer Guide</h1>

This guide applies to the development within the OpenSearch-py-ml project 

- [Getting started guide](#getting-started-guide)
  - [Key technologies](#key-technologies)
  - [Prerequisites](#prerequisites)
  - [Fork and clone OpenSearch-py-ml](#fork-and-clone-opensearch-py-ml)
  - [Install OpenSearch-py-ml dependencies](#Install-Opensearch-py-ml-dependencies)
  - [Run OpenSearch](#run-opensearch)
  - [Next Steps](#next-steps)
- [Code guidelines](#code-guidelines)


## Getting started guide

This guide is for any developer who wants a running local development environment where you can make, see, and test changes. It's opinionated to get you running as quickly and easily as possible, but it's not the only way to set up a development environment.

If you're only interested in installing and running this project, you can install from [pypi](https://pypi.org/project/opensearch-py-ml/)

If you're planning to contribute code (features or fixes) to this repository, great! Make sure to also read the [contributing guide](CONTRIBUTING.md).

### Key technologies

OpenSearch-py-ml is primarily a python based client plugin for machine learning in opensearch. To effectively contribute you should be familiar with PYTHON.

### Prerequisites

To develop on OpenSearch-py-ml, you'll need:

- A [GitHub account](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account)
- [`git`](https://git-scm.com/) for version control
- [`Python`](https://www.python.org/), Package installer for python. for example:[`pip`](https://pypi.org/project/pip/)
- A code editor of your choice, configured for Python. If you don't have a favorite editor, we suggest [Pycharm](https://www.jetbrains.com/pycharm/)

If you already have these installed or have your own preferences for installing them, skip ahead to the [Fork and clone OpenSearch-py-ml](#fork-and-clone-opensearch-py-ml) section.

#### Install `git`

If you don't already have it installed (check with `git --version`) we recommend following [the `git` installation guide for your OS](https://git-scm.com/downloads).

#### Install `python`

You can install any version of python starting from 3.8.

### Fork and clone OpenSearch-py-ml

All local development should be done in a [forked repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
Fork OpenSearch-py-ml by clicking the "Fork" button at the top of the [GitHub repository](https://github.com/opensearch-project/OpenSearch-py-ml).

Clone your forked version of OpenSearch-py-ml to your local machine (replace `opensearch-project` in the command below with your GitHub username):

```bash
$ git clone git@github.com:opensearch-project/opensearch-py-ml.git
```

### Install Opensearch-py-ml dependencies

If you haven't already, change directories to your cloned repository directory:

```bash
$ cd OpenSearch-py-ml
```

The `pip install` command will install the project's dependencies and build all internal packages and plugins.

```bash
$ pip install -r requirements-dev.txt
```


### Run OpenSearch

OpenSearch-py-ml requires a running version of OpenSearch (from opensearch 2.5) to connect to. 

You can install opensearch multiple ways:

1. https://opensearch.org/downloads.html#docker-compose
2. https://opensearch.org/docs/2.5/install-and-configure/install-opensearch/tar/


### Next Steps

Now that you have a development environment to play with, there are a number of different paths you may take next.

#### Update the settings for the cluster
After navigating to OpenSearch Dashboards you should update the persistent settings for the cluster. The settings will update the behavior of the machine learning plugin, specifically the ml_commons plugin. ML Commons cluster settings: https://opensearch.org/docs/latest/ml-commons-plugin/cluster-settings/

You should paste this settings in the `Dev Tools` window and run it:

```yml
 PUT /_cluster/settings
 {
   "persistent" : {
     "plugins.ml_commons.only_run_on_ml_node" : false, 
     "plugins.ml_commons.native_memory_threshold" : 100, 
     "plugins.ml_commons.max_model_on_node": 20,
     "plugins.ml_commons.enable_inhouse_python_model": true
   }
 }
```

#### Review user tutorials to understand the key features and workflows

- These [Notebook Examples](https://opensearch-project.github.io/opensearch-py-ml/examples/index.html) will show you how to use opensearch-py-ml for data exploration and machine learning.
- [API references](https://opensearch-project.github.io/opensearch-py-ml/reference/index.html) provides helpful guidance using different functionalities of opensearch-py-ml

#### To test code formatting and linting issue we can run

```bash
$ nox -s lint
```

#### To fix code formatting and linting issue we can run

```bash
$ nox -s format
```

#### To run tests

```bash
$ nox -s test
```

#### To test documentation

```bash
# New HTML pages will be created in build/html
$ cd docs
$ pip install -r requirements-docs.txt
$ make clean
$ make html
```


#### Default setup for opensearch

```yml
opensearch.hosts: ["https://localhost:9200"]
opensearch.username: "admin" # Default username
opensearch.password: "admin" # Default password
```

## Code guidelines

#### Filenames

All filenames should use `snake_case`.

**Right:** `opensearch_py_ml/ml_commons/ml_commons_client.py`

**Wrong:** `opensearch_py_ml/mlCommons/mlCommonsClient.py`

#### Do not comment out code

We use a version management system. If a line of code is no longer needed,
remove it, don't simply comment it out.

#### Avoid magic numbers/strings

These are numbers (or other values) simply used in line in your code. _Do not
use these_, give them a variable name, so they can be understood and changed
easily.

```python
// good
minWidth = 300

if width < minWidth:
  ...

// bad
if width < 300:
  ...
```

#### Avoid global definitions

Don't do this. Everything should be wrapped in a module that can be depended on
by other modules. Even things as simple as a single value should be a module.

#### Write small functions

Keep your functions short. A good function fits on a slide that the people in
the last row of a big room can comfortably read. So don't count on them having
perfect vision and limit yourself to ~15 lines of code per function.

