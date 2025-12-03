# CHANGELOG
Inspired from [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [1.4.0]

### Added

### Changed
- Update semantic highlighting sagemaker endpoint deploy scripts ([#585](https://github.com/opensearch-project/opensearch-py-ml/pull/585))

### Fixed

## [1.3.0]

### Added
- Add space type mapping for sentence transformer models by @nathaliellenaa in ([#512](https://github.com/opensearch-project/opensearch-py-ml/pull/512))
- Add example script for deploying semantic highlighter model on aws sagemaker. ([#513](https://github.com/opensearch-project/opensearch-py-ml/pull/513))
- Add model_init_kwargs to sparse model uploading pipeline. ([#555](https://github.com/opensearch-project/opensearch-py-ml/pull/555))
// Move to 1.4.0
- Add agentic memory integration script ([#592](https://github.com/opensearch-project/opensearch-py-ml/pull/592))

### Changed
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#415](https://github.com/opensearch-project/opensearch-py-ml/pull/415))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#483](https://github.com/opensearch-project/opensearch-py-ml/pull/483))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#485](https://github.com/opensearch-project/opensearch-py-ml/pull/485))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#489](https://github.com/opensearch-project/opensearch-py-ml/pull/489))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#492](https://github.com/opensearch-project/opensearch-py-ml/pull/492))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#494](https://github.com/opensearch-project/opensearch-py-ml/pull/494))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#496](https://github.com/opensearch-project/opensearch-py-ml/pull/496))
- Update model upload history -  opensearch-project/opensearch-semantic-highlighter-v1 (v.1.0.0)(TORCH_SCRIPT) by @greaa-aws ([#499](https://github.com/opensearch-project/opensearch-py-ml/pull/499))
- Update model upload history -  opensearch-project/opensearch-semantic-highlighter-v1 (v.1.0.0)(TORCH_SCRIPT) by @b4sjoo ([#503](https://github.com/opensearch-project/opensearch-py-ml/pull/503))
- Update model upload history -  sentence-transformers/all-MiniLM-L6-v2 (v.1.0.2)(BOTH) by @nathaliellenaa ([#523](https://github.com/opensearch-project/opensearch-py-ml/pull/523))
- Update model upload history -  sentence-transformers/all-MiniLM-L12-v2 (v.1.0.2)(BOTH) by @nathaliellenaa ([#526](https://github.com/opensearch-project/opensearch-py-ml/pull/526))
- Update model upload history -  sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (v.1.0.2)(BOTH) by @nathaliellenaa ([#528](https://github.com/opensearch-project/opensearch-py-ml/pull/528))
- Update model upload history -  sentence-transformers/paraphrase-MiniLM-L3-v2 (v.1.0.2)(BOTH) by @nathaliellenaa ([#530](https://github.com/opensearch-project/opensearch-py-ml/pull/530))
- Update model upload history -  sentence-transformers/msmarco-distilbert-base-tas-b (v.1.0.3)(BOTH) by @nathaliellenaa ([#532](https://github.com/opensearch-project/opensearch-py-ml/pull/532))
- Update model upload history -  sentence-transformers/all-distilroberta-v1 (v.1.0.2)(BOTH) by @nathaliellenaa ([#535](https://github.com/opensearch-project/opensearch-py-ml/pull/535))
- Update model upload history -  sentence-transformers/all-mpnet-base-v2 (v.1.0.2)(BOTH) by @nathaliellenaa ([#537](https://github.com/opensearch-project/opensearch-py-ml/pull/537))
- Update model upload history -  sentence-transformers/multi-qa-mpnet-base-dot-v1 (v.1.0.2)(BOTH) by @nathaliellenaa ([#539](https://github.com/opensearch-project/opensearch-py-ml/pull/539))
- Update model upload history -  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (v.1.0.2)(BOTH) by @nathaliellenaa ([#541](https://github.com/opensearch-project/opensearch-py-ml/pull/541))
- Update model upload history -  sentence-transformers/paraphrase-mpnet-base-v2 (v.1.0.1)(BOTH) by @nathaliellenaa ([#543](https://github.com/opensearch-project/opensearch-py-ml/pull/543))
- Update model upload history -  sentence-transformers/distiluse-base-multilingual-cased-v1 (v.1.0.2)(TORCH_SCRIPT) by @nathaliellenaa ([#545](https://github.com/opensearch-project/opensearch-py-ml/pull/545))
- Update pretrained_models_all_versions.json (2025-06-02 17:57:40) by @nathaliellenaa ([#547](https://github.com/opensearch-project/opensearch-py-ml/pull/547))
- Update pretrained_models_all_versions.json by @nathaliellenaa ([#548](https://github.com/opensearch-project/opensearch-py-ml/pull/548))
- Update model upload history -  opensearch-project/opensearch-semantic-highlighter-v1 (v.1.0.0)(TORCH_SCRIPT) by @nathaliellenaa ([#550](https://github.com/opensearch-project/opensearch-py-ml/pull/550))
- Update pretrained_models_all_versions.json by @nathaliellenaa ([#560](https://github.com/opensearch-project/opensearch-py-ml/pull/560))
- Upgrade torch version to fix security issue ([#561](https://github.com/opensearch-project/opensearch-py-ml/pull/561))
- Upgrade numpy and python version ([#562](https://github.com/opensearch-project/opensearch-py-ml/pull/562))
- Update space type mapping for sentence transformer models ([#574](https://github.com/opensearch-project/opensearch-py-ml/pull/574))

### Fixed
- Fix for uploading models with function_name instead of model_task ([#553](https://github.com/opensearch-project/opensearch-py-ml/pull/553))

## [1.2.0]

### Added
- Add workflows and scripts for automating model tracing and uploading process by @thanawan-atc in ([#209](https://github.com/opensearch-project/opensearch-py-ml/pull/209))
- Add workflow and scripts for automating model listing updating process by @thanawan-atc in ([#210](https://github.com/opensearch-project/opensearch-py-ml/pull/210))
- Add script to trigger ml-models-release jenkins workflow with generic webhook by @thanawan-atc in ([#211](https://github.com/opensearch-project/opensearch-py-ml/pull/211))
- Add example notebook for tracing and registering a CLIPTextModel to OpenSearch with the Neural Search plugin by @patrickbarnhart in ([#283](https://github.com/opensearch-project/opensearch-py-ml/pull/283))
- Add support for train api functionality by @rawwar in ([#310](https://github.com/opensearch-project/opensearch-py-ml/pull/310))
- Add support for Model Access Control - Register, Update, Search and Delete by @rawwar in ([#332](https://github.com/opensearch-project/opensearch-py-ml/pull/332))
- Add support for model connectors by @rawwar in ([#345](https://github.com/opensearch-project/opensearch-py-ml/pull/345))
- Add support for model profiles by @rawwar in ([#358](https://github.com/opensearch-project/opensearch-py-ml/pull/358))
- Support for security default admin credential changes in 2.12.0 in ([#365](https://github.com/opensearch-project/opensearch-py-ml/pull/365))
- adding cross encoder models in the pre-trained traced list ([#378](https://github.com/opensearch-project/opensearch-py-ml/pull/378))
- Add workflows and scripts for sparse encoding model tracing and uploading process by @conggguan in ([#394](https://github.com/opensearch-project/opensearch-py-ml/pull/394))
- Implemented `predict` method and added unit tests by @yerzhaisang([425](https://github.com/opensearch-project/opensearch-py-ml/pull/425))
- Add workflows and scripts for sparse tokenizer model tracing and uploading process; enhance sparse encoding tracing with custom params by @zhichao-aws in ([#459](https://github.com/opensearch-project/opensearch-py-ml/pull/459))
- Added Connector CLI into `opensearch-py-ml` by @nathaliellenaa in ([#461](https://github.com/opensearch-project/opensearch-py-ml/pull/461))
- Add semantic highlighter model auto tracing support ([#486](https://github.com/opensearch-project/opensearch-py-ml/pull/486))
- Add semantic highlighter model to pre-trained model list ([#504](https://github.com/opensearch-project/opensearch-py-ml/pull/504))
- Fix jenkins token in model_listing_uploader.yml CI ([#505](https://github.com/opensearch-project/opensearch-py-ml/pull/505))
- Add documentation for CLI feature by @nathaliellenaa in ([#476](https://github.com/opensearch-project/opensearch-py-ml/pull/476))

### Changed
- Add a parameter for customize the upload folder prefix ([#398](https://github.com/opensearch-project/opensearch-py-ml/pull/398))
- Modify ml-models.JenkinsFile so that it takes model format into account and can be triggered with generic webhook by @thanawan-atc in ([#211](https://github.com/opensearch-project/opensearch-py-ml/pull/211))
- Update demo_tracing_model_torchscript_onnx.ipynb to use make_model_config_json by @thanawan-atc in ([#220](https://github.com/opensearch-project/opensearch-py-ml/pull/220))
- Bump torch from 1.13.1 to 2.0.1 and add onnx dependency by @thanawan-atc ([#237](https://github.com/opensearch-project/opensearch-py-ml/pull/237))
- Update pretrained_model_listing.json (2023-08-23 16:51:21) by @dhrubo-os ([#248](https://github.com/opensearch-project/opensearch-py-ml/pull/248))
- Store new format of model listing at pretrained_models_all_versions.json instead of pre_trained_models.json in S3 and pretrained_model_listing.json in repo by @thanawan-atc ([#256](https://github.com/opensearch-project/opensearch-py-ml/pull/256))
- Make the model tracing-uploading-releasing workflow fail early for ≥2GB model by @thanawan-atc ([#258](https://github.com/opensearch-project/opensearch-py-ml/pull/258))
- Rename model/model-listing workflows by @thanawan-atc ([#260](https://github.com/opensearch-project/opensearch-py-ml/pull/260))
- Update pretrained_models_all_versions.json (2023-08-30 14:07:38) by @dhrubo-os ([#264](https://github.com/opensearch-project/opensearch-py-ml/pull/264))
- Update model upload history -  sentence-transformers/msmarco-distilbert-base-tas-b (v.1.0.2)(BOTH) by @dhrubo-os ([#272](https://github.com/opensearch-project/opensearch-py-ml/pull/272))
- Enable model listing workflow to exclude old 1.0.0 models from the generated model listing by @thanawan-atc ([#265](https://github.com/opensearch-project/opensearch-py-ml/pull/265))
- Have model upload workflow require approval from two code maintainers by @thanawan-atc ([#273](https://github.com/opensearch-project/opensearch-py-ml/pull/273))
- Update pretrained_models_all_versions.json (2023-09-08 13:14:07) by @dhrubo-os ([#277](https://github.com/opensearch-project/opensearch-py-ml/pull/277))
- Update model upload history -  sentence-transformers/distiluse-base-multilingual-cased-v1 (v.1.0.1)(TORCH_SCRIPT) by @dhrubo-os ([#281](https://github.com/opensearch-project/opensearch-py-ml/pull/281))
- Update pretrained_models_all_versions.json (2023-09-14 10:28:41) by @dhrubo-os ([#282](https://github.com/opensearch-project/opensearch-py-ml/pull/282))
- Enable the model upload workflow to add model_content_size_in_bytes & model_content_hash_value to model config automatically @thanawan-atc ([#291](https://github.com/opensearch-project/opensearch-py-ml/pull/291))
- Update pretrained_models_all_versions.json (2023-10-18 18:11:34) by @dhrubo-os ([#322](https://github.com/opensearch-project/opensearch-py-ml/pull/322))
- Update model upload history -  sentence-transformers/paraphrase-mpnet-base-v2 (v.1.0.0)(BOTH) by @dhrubo-os ([#321](https://github.com/opensearch-project/opensearch-py-ml/pull/321))
- Replaced usage of `is_datetime_or_timedelta_dtype` with `is_timedelta64_dtype` and `is_datetime64_any_dtype` by @rawwar ([#316](https://github.com/opensearch-project/opensearch-py-ml/pull/316))
- use try-except-else block for handling unexpected exceptions during integration tests by @rawwar([#370](https://github.com/opensearch-project/opensearch-py-ml/pull/370))
- Removed pandas version pin in nox tests by @rawwar ([#368](https://github.com/opensearch-project/opensearch-py-ml/pull/368))
- Switch AL2 to AL2023 agent and DockerHub to ECR images in ml-models.JenkinsFile ([#377](https://github.com/opensearch-project/opensearch-py-ml/pull/377))
- Refactored validators in ML Commons' client([#385](https://github.com/opensearch-project/opensearch-py-ml/pull/385))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#407](https://github.com/opensearch-project/opensearch-py-ml/pull/407))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#405](https://github.com/opensearch-project/opensearch-py-ml/pull/405))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-v2-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#410](https://github.com/opensearch-project/opensearch-py-ml/pull/410))
- updating listing file with three v2 sparse model - by @dhrubo-os ([#412](https://github.com/opensearch-project/opensearch-py-ml/pull/412))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#417](https://github.com/opensearch-project/opensearch-py-ml/pull/417))
- Update model upload history -  opensearch-project/opensearch-neural-sparse-encoding-v2-distill (v.1.0.0)(TORCH_SCRIPT) by @dhrubo-os ([#419](https://github.com/opensearch-project/opensearch-py-ml/pull/419))
- Upgrade GitHub Actions workflows to use `@v4` to prevent deprecation issues with `@v3` by @yerzhaisang ([#428](https://github.com/opensearch-project/opensearch-py-ml/pull/428))
- Bump pandas from 1.5.3 to the latest stable version by @yerzhaisang ([#422](https://github.com/opensearch-project/opensearch-py-ml/pull/422))
- Upgrade mypy, sphinx, sphinx-rtd-theme, and multiple GitHub Actions (setup-python, backport, codecov-action, create-pull-request, get-pr-commits) by @yerzhaisang([#437](https://github.com/opensearch-project/opensearch-py-ml/pull/437))
- Increment version to 1.2.0 and update maintainer list in _version.py by @nathaliellenaa([#516](https://github.com/opensearch-project/opensearch-py-ml/pull/516))

### Fixed
- Fix the wrong final zip file name in model_uploader workflow, now will name it by the upload_prefix alse.([#413](https://github.com/opensearch-project/opensearch-py-ml/pull/413/files))
- Fix the wrong input parameter for model_uploader's base_download_path in jekins trigger.([#402](https://github.com/opensearch-project/opensearch-py-ml/pull/402))
- Enable make_model_config_json to add model description to model config file by @thanawan-atc in ([#203](https://github.com/opensearch-project/opensearch-py-ml/pull/203))
- Correct demo_ml_commons_integration.ipynb by @thanawan-atc in ([#208](https://github.com/opensearch-project/opensearch-py-ml/pull/208))
- Handle the case when the model max length is undefined in tokenizer by @thanawan-atc in ([#219](https://github.com/opensearch-project/opensearch-py-ml/pull/219))
- Change to use tokenCredentialId for triggering ml-models-release via generic webhook by @thanawan-atc in ([#240](https://github.com/opensearch-project/opensearch-py-ml/pull/240))
- Fix typo in model/model-listing workflows by @thanawan-atc in ([#244](https://github.com/opensearch-project/opensearch-py-ml/pull/244))
- Fix typo in model-listing workflows by @thanawan-atc in ([#246](https://github.com/opensearch-project/opensearch-py-ml/pull/246))
- Add BUCKET_NAME to ml-models-release jenkinsfile by @thanawan-atc in ([#249](https://github.com/opensearch-project/opensearch-py-ml/pull/249))
- Roll over pretrained_model_listing.json because of ml-commons dependency by @thanawan-atc in ([#252](https://github.com/opensearch-project/opensearch-py-ml/pull/252))
- Fix pandas dependency issue in nox session by installing pandas package to python directly by @thanawan-atc in ([#266](https://github.com/opensearch-project/opensearch-py-ml/pull/266))
- Fix conditional job execution issue in model upload workflow by @thanawan-atc in ([#294](https://github.com/opensearch-project/opensearch-py-ml/pull/294))
- fix bug in `MLCommonClient_client.upload_model` by @rawwar in ([#336](https://github.com/opensearch-project/opensearch-py-ml/pull/336))
- fix lint issues on main by @rawwar in ([#374](https://github.com/opensearch-project/opensearch-py-ml/pull/374))
- fix CVE vulnerability by @rawwar in ([#383](https://github.com/opensearch-project/opensearch-py-ml/pull/383))
- refactor: replace 'payload' with 'body' in `create_standalone_connector` by @yerzhaisang ([#424](https://github.com/opensearch-project/opensearch-py-ml/pull/424))
- Fix CVE vulnerability by @nathaliellenaa in ([#447](https://github.com/opensearch-project/opensearch-py-ml/pull/447))
- Use one model id for sparse model integ test to reduce flaky error caused by throttling ([#475](https://github.com/opensearch-project/opensearch-py-ml/pull/475))
- Fix sparse upload workflow caused by SDPA exception ([#478](https://github.com/opensearch-project/opensearch-py-ml/pull/478))

## [1.1.0]

### Added

- Adding documentation for model group id @dhrubo-os ([#176](https://github.com/opensearch-project/opensearch-py-ml/pull/176))
- listing pre-trained release models @dhrubo-os ([#85](https://github.com/opensearch-project/opensearch-py-ml/pull/85))
- Upload pretrained models @AlibiZhenis ([#111](https://github.com/opensearch-project/opensearch-py-ml/pull/111/files))
- Added delete task API. @Nurlanprog ([#127](https://github.com/opensearch-project/opensearch-py-ml/pull/127/files))
- Add test coverage statistics to Codecod @Yerzhaisang @bl1nkker ([#138](https://github.com/opensearch-project/opensearch-py-ml/pull/138/files))
- Merging Feature/mcorr to main @dhrubo-os @greaa-aws ([#150](https://github.com/opensearch-project/opensearch-py-ml/pull/150/files))
- Adding boolean argument to ML Commons API @AlibiZhenis ([#143](https://github.com/opensearch-project/opensearch-py-ml/pull/143/files))
- Rename APIs for model serving framework @AlibiZhenis ([#159](https://github.com/opensearch-project/opensearch-py-ml/pull/159/files))
- Adding executeAPI @AlibiZhenis ([#165](https://github.com/opensearch-project/opensearch-py-ml/pull/165/files))
- Adding documentation for model group id @dhrubo-os ([#176](https://github.com/opensearch-project/opensearch-py-ml/pull/176/files))
- Search task added @Nurlanprog ([#177](https://github.com/opensearch-project/opensearch-py-ml/pull/177/files))
- adding jupyter notebook based documentation for metrics correlation algorithm by @AlibiZhenis ([#186](https://github.com/opensearch-project/opensearch-py-ml/pull/186))

### Changed
- Update jenkins file to use updated docker image ([#189](https://github.com/opensearch-project/opensearch-py-ml/pull/189))
- Updated documentation @dhrubo-os ([#98](https://github.com/opensearch-project/opensearch-py-ml/pull/98))
- Updating ML Commons API documentation @AlibiZhenis ([#156](https://github.com/opensearch-project/opensearch-py-ml/pull/156))

### Fixed
- Fix ModelUploader bug & Update model tracing demo notebook by @thanawan-atc in ([#185](https://github.com/opensearch-project/opensearch-py-ml/pull/185))
- Fix make_model_config_json function by @thanawan-atc in ([#188](https://github.com/opensearch-project/opensearch-py-ml/pull/188))
- Make make_model_config_json function more concise by @thanawan-atc in ([#191](https://github.com/opensearch-project/opensearch-py-ml/pull/191))
- Enabled auto-truncation for any pretrained models by @Yerzhaisang in ([#192](https://github.com/opensearch-project/opensearch-py-ml/pull/192))
- Generalize make_model_config_json function by @thanawan-atc in ([#200](https://github.com/opensearch-project/opensearch-py-ml/pull/200))

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

### Deprecated

### Removed
- Removing os_client dependency + added 2.4.0 version for integration test by @dhrubo-os in ([#45](https://github.com/opensearch-project/opensearch-py-ml/pull/45))

### Fixed
- Fixed bugs in Training Script
- Fix file extension issue and add wait for multi-processes  by @mingshl in ([#42](https://github.com/opensearch-project/opensearch-py-ml/pull/42))
- Fixing train documentation by @dhrubo-os in ([#44](https://github.com/opensearch-project/opensearch-py-ml/pull/44))
- Upgrade package version to fix security issues and format code   by @mingshl in ([#51](https://github.com/opensearch-project/opensearch-py-ml/pull/51))
- Bug fix of SentenceTransformerModel + add integration test from model by @dhrubo-os in ([#63](https://github.com/opensearch-project/opensearch-py-ml/pull/63))

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
[1.1.0]: https://github.com/opensearch-project/opensearch-py-ml/commits/1.1.0
[1.2.0]: https://github.com/opensearch-project/opensearch-py-ml/commits/1.2.0
[1.3.0]: https://github.com/opensearch-project/opensearch-py-ml/commits/1.3.0
