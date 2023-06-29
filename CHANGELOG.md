# CHANGELOG
Inspired from [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [1.1.0]

### Added

- adding documentation for model group id @dhrubo-os ([#176](https://github.com/opensearch-project/opensearch-py-ml/pull/176))

## [1.0.0]    

### Added
- Added multiple notebooks in documentation for better clarification
- Added integration tests and more functionalities for MLCommons integration
- Added support for tracing model in Onnx format
- Add make_model_config function and add doc by @mingshl ([#46](https://github.com/opensearch-project/opensearch-py-ml/pull/46))
- Unit test for SentenceTransformerModel by @mingshl in ([#52](https://github.com/opensearch-project/opensearch-py-ml/pull/52))
- Notebook_documentation by @dhrubo-os in ([#66]https://github.com/opensearch-project/opensearch-py-ml/pull/66)
- Added download link to the notebook by @dhrubo-os in ([#73](https://github.com/opensearch-project/opensearch-py-ml/pull/73))

### Changed
- Updating installation instruction by @dhrubo-os in ([#41](https://github.com/opensearch-project/opensearch-py-ml/pull/41))
- Refactoring upload_api + added integration test + added load model API by @dhrubo-os in ([#54](https://github.com/opensearch-project/opensearch-py-ml/pull/54))
- Updated MAINTAINERS.md format. by @dblock in ([#64]https://github.com/opensearch-project/opensearch-py-ml/pull/64)
- Merged generate.py demo notebook with training notebook by @dhrubo-os in ([#67](https://github.com/opensearch-project/opensearch-py-ml/pull/67))
- Tracing model with onnx format + changed OpenSearch version to 2.5.0 ([#69](https://github.com/opensearch-project/opensearch-py-ml/pull/69))
- Updating docs workflow to 2.5.0 also by @dhrubo-os([#71](https://github.com/opensearch-project/opensearch-py-ml/pull/71))
- Update notebook + version update by @dhrubo-os in ([#76](https://github.com/opensearch-project/opensearch-py-ml/pull/76))
- Update model tracing demo notebook by @thanawan-atc in ([#185](https://github.com/opensearch-project/opensearch-py-ml/pull/185))

### Deprecated

### Removed
- Removing os_client dependency + added 2.4.0 version for integration test by @dhrubo-os in ([#45](https://github.com/opensearch-project/opensearch-py-ml/pull/45))

### Fixed
- Fixed bugs in Training Script
- Fix file extension issue and add wait for multi-processes  by @mingshl in ([#42](https://github.com/opensearch-project/opensearch-py-ml/pull/42))
- Fixing train documentation by @dhrubo-os in ([#44](https://github.com/opensearch-project/opensearch-py-ml/pull/44))
- Upgrade package version to fix security issues and format code   by @mingshl in ([#51](https://github.com/opensearch-project/opensearch-py-ml/pull/51))
- Bug fix of SentenceTransformerModel + add integration test from model by @dhrubo-os in ([#63](https://github.com/opensearch-project/opensearch-py-ml/pull/63))
- Bug fix of ModelUploader by @thanawan-atc in ([#185](https://github.com/opensearch-project/opensearch-py-ml/pull/185))

### Security
- Bump opensearch-py from 2.0.1 to 2.1.1 by @dependabot ([#70](https://github.com/opensearch-project/opensearch-py-ml/pull/70))


## [1.0.0b1]

### Added
- Merging all the codes of dev to main branch. by @dhrubo-os in ([#9](https://github.com/opensearch-project/opensearch-py-ml/pull/9))
- Custom model upload + some documenation. by @dhrubo-os in ([#10]https://github.com/opensearch-project/opensearch-py-ml/pull/10)
- Adding template files to make repo public by @dhrubo-os in ([#12]https://github.com/opensearch-project/opensearch-py-ml/pull/12)
- Linting for windows and mac by @dhrubo-os in ([#14]https://github.com/opensearch-project/opensearch-py-ml/pull/14)
- Add semantic search training script by @mingshl in ([#18]https://github.com/opensearch-project/opensearch-py-ml/pull/18)
- Added documentation for ml_commons by @dhrubo-os in ([#23]https://github.com/opensearch-project/opensearch-py-ml/pull/23)
- Add release workflows by @gaiksaya in ([#27]https://github.com/opensearch-project/opensearch-py-ml/pull/27)
- Add demo notebook and return model by @mingshl in ([#34]https://github.com/opensearch-project/opensearch-py-ml/pull/34)

### Changed
- Added documentation + fixed ipynb files by @dhrubo-os in ([#19]https://github.com/opensearch-project/opensearch-py-ml/pull/19)
- Changed naming + fixed ci workflow by @dhrubo-os in ([#21]https://github.com/opensearch-project/opensearch-py-ml/pull/21)
- Updating version by @dhrubo-os in ([#33]https://github.com/opensearch-project/opensearch-py-ml/pull/33)
- Change version + documentation by @dhrubo-os in ([#36]https://github.com/opensearch-project/opensearch-py-ml/pull/36)
- Change version to 1.0.0b1 by @ylwu-amzn in ([#38]https://github.com/opensearch-project/opensearch-py-ml/pull/38)

### Deprecated

### Removed
- Removing installation guide + adding more md files by @dhrubo-os in ([#26]https://github.com/opensearch-project/opensearch-py-ml/pull/26)

### Fixed
- Fixing documentation issue by @dhrubo-os in ([#20]https://github.com/opensearch-project/opensearch-py-ml/pull/20)
- Increment jenkins lib version and fix GHA job name by @gaiksaya in ([#37]https://github.com/opensearch-project/opensearch-py-ml/pull/37)

### Security


[1.0.0]: https://github.com/opensearch-project/opensearch-py-ml/compare/1.0.0b1...1.0.0
[1.0.0b1]: https://github.com/opensearch-project/opensearch-py-ml/commits/1.0.0b1