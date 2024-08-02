# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import json
import os
from abc import ABC, abstractmethod
from zipfile import ZipFile

import requests

from opensearch_py_ml.ml_commons.ml_common_utils import (
    LICENSE_URL,
    SPARSE_ENCODING_FUNCTION_NAME,
)


class BaseUploadModel(ABC):
    """
    A base class for uploading models to OpenSearch pretrained model hub.
    """

    def __init__(
        self, model_id: str, folder_path: str = None, overwrite: bool = False
    ) -> None:
        self.model_id = model_id
        self.folder_path = folder_path
        self.overwrite = overwrite

    @abstractmethod
    def save_as_pt(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_as_onnx(self, *args, **kwargs):
        pass

    @abstractmethod
    def make_model_config_json(
        self,
        version_number: str,
        model_format: str,
        description: str,
    ) -> str:
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


class SparseModel(BaseUploadModel, ABC):
    """
    Class for autotracing the Sparse Encoding model.
    """

    def __init__(
        self,
        model_id: str,
        folder_path: str = "./model_files/",
        overwrite: bool = False,
    ):
        super().__init__(model_id, folder_path, overwrite)
        self.model_id = model_id
        self.folder_path = folder_path
        self.overwrite = overwrite
        self.function_name = SPARSE_ENCODING_FUNCTION_NAME

    def pre_process(self):
        pass

    def post_process(self):
        pass
