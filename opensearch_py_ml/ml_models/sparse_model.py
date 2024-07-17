# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from zipfile import ZipFile

import requests

from .ml_commons_utils import LICENSE_URL


class SparseModel:
    """
    Class for autotracing the Sparse Encoding model.
    """

    def __init__(
        self,
        model_id: str,
        folder_path: str = "./model_files/",
        overwrite: bool = False,
    ):
        self.model_id = model_id
        self.folder_path = folder_path
        self.overwrite = overwrite

    def pro_process(self):
        pass

    def post_process(self):
        pass

    def _add_apache_license_to_zip(self, zip_file_path: str):
        with ZipFile(zip_file_path, "a") as zipObj, requests.get(LICENSE_URL) as r:
            assert r.status_code == 200, "Failed to download license"
            zipObj.writestr("LICENSE", r.content)

    def save_as_pt(
        self,
        model_id,
        sentences,
        add_apache_license=True,
    ):
        pass

    def save_as_onnx(
        self,
        model_id,
        add_apache_license=True,
    ):
        pass

    def make_model_config_json(
        self,
        version_number: str = "1.0",
        model_format: str = "TORCH_SCRIPT",
        description: str = None,
    ) -> str:
        pass

    def save(self, path: str):
        pass

    def _fill_null_truncation_field(
        self,
        save_json_folder_path: str,
        max_length: int,
    ) -> None:
        """
        Fill truncation field in tokenizer.json when it is null

        :param save_json_folder_path:
             path to save model json file, e.g, "home/save_pre_trained_model_json/")
        :type save_json_folder_path: string
        :param max_length:
             maximum sequence length for model
        :type max_length: int
        :return: no return value expected
        :rtype: None
        """
        tokenizer_file_path = os.path.join(save_json_folder_path, "tokenizer.json")
        with open(tokenizer_file_path) as user_file:
            parsed_json = json.load(user_file)
        if "truncation" not in parsed_json or parsed_json["truncation"] is None:
            parsed_json["truncation"] = {
                "direction": "Right",
                "max_length": max_length,
                "strategy": "LongestFirst",
                "stride": 0,
            }
            with open(tokenizer_file_path, "w") as file:
                json.dump(parsed_json, file, indent=2)

    def _add_apache_license_to_model_zip_file(self, model_zip_file_path: str):
        """
        Add Apache-2.0 license file to the model zip file at model_zip_file_path

        :param model_zip_file_path:
            Path to the model zip file
        :type model_zip_file_path: string
        :return: no return value expected
        :rtype: None
        """
        r = requests.get(LICENSE_URL)
        assert r.status_code == 200, "Failed to add license file to the model zip file"

        with ZipFile(str(model_zip_file_path), "a") as zipObj:
            zipObj.writestr("LICENSE", r.content)
