# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
**Repository:** <https://github.com/elastic/eland>

We internally develop using the PyCharm IDE. For PyCharm, we are
currently using a minimum version of PyCharm 2019.2.4.

### Configuring PyCharm And Running Tests

(All commands should be run from module root)

* Create a new project via \'Check out from Version

    Control\'-\>\'Git\' on the \"Welcome to PyCharm\" page (or other)

* Enter the URL to your fork of eland

    (e.g.  `git@github.com:stevedodson/opensearch_py_ml.git` )

* Click \'Yes\' for \'Checkout from Version Control\'
* Configure PyCharm environment:
* In \'Preferences\' configure a \'Project: eland\'-\>\'Project

    Interpreter\'. Generally, we recommend creating a virtual
    environment (TODO link to installing for python version support).

* In \'Preferences\' set \'Tools\'-\>\'Python Integrated

    Tools\'-\>\'Default test runner\' to `pytest`

* In \'Preferences\' set \'Tools\'-\>\'Python Integrated

    Tools\'-\>\'Docstring format\' to `numpy`

* To install development requirements. Open terminal in virtual environment and run

    ``` bash
    > pip install -r requirements-dev.txt
    ```

* Setup Elasticsearch instance with docker

    ``` bash
    > ELASTICSEARCH_VERSION=elasticsearch:7.x-SNAPSHOT .ci/run-elasticsearch.sh
    ```

* Now check `http://localhost:9200`
* Install local `eland` module (required to execute notebook tests)

    ``` bash
    > python setup.py install
    ```

* To setup test environment:

    ``` bash
    > python -m tests.setup_tests
    ```

    (Note this modifies Elasticsearch indices)

* To validate installation, open python console and run

    ``` bash
    > import opensearch_py_ml as ed
    > ed_df = ed.DataFrame('localhost', 'flights')
    ```

* To run the automatic formatter and check for lint issues run

    ``` bash
    > nox -s format
    ```

* To test specific versions of Python run

    ``` bash
    > nox -s test-3.8
    ```

### Documentation

* [Install pandoc on your system](https://pandoc.org/installing.html) . For Ubuntu or Debian you can do

    ``` bash
    > sudo apt-get install -y pandoc
    ```

* Install documentation requirements. Open terminal in virtual environment and run

    ``` bash
    > pip install -r docs/requirements-docs.txt
    ```

* To verify/generate documentation run

    ``` bash
    > nox -s docs
    ```
