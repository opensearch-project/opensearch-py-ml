# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.gcp_model import GCPModel


class TestGCPModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.gcp_model = GCPModel()
        self.project_id = "test_project_id"
        self.model_id = "test_model_id"
        self.access_token = "test_access_token"

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_vertexai_embedding(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a GCP connector with VertexAI embedding model"""
        # Set mock return values
        mock_get_model_details.return_value = "1"

        result = self.gcp_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="VertexAI embedding model",
            project_id=self.project_id,
            model_id=self.model_id,
            access_token=self.access_token,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://.*-aiplatform\\.googleapis\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Google Cloud Platform", GCPModel.OPEN_SOURCE, "VertexAI embedding model"
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_custom_model(
        self,
        mock_get_model_details,
        mock_set_trusted_endpoint,
        mock_custom_model,
    ):
        """Test creating an GCP connector with custom model"""
        result = self.gcp_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Custom model",
            project_id=self.project_id,
            model_id=self.model_id,
            access_token=self.access_token,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://.*-aiplatform\\.googleapis\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Google Cloud Platform", GCPModel.OPEN_SOURCE, "Custom model"
        )
        mock_custom_model.assert_called_once()
        self.assertTrue(result)

    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "test_project_id",
            "test_model_id",
            "test_access_token",
        ],
    )
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        result = self.gcp_model.create_connector(
            helper=self.mock_helper, save_config_method=self.mock_save_config
        )
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating an GCP connector with invalid model choice"""
        self.gcp_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Invalid Model",
            project_id=self.project_id,
            model_id=self.model_id,
            access_token=self.access_token,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once()

    def test_create_connector_failure(self):
        """Test creating an GCP connector in failure scenario"""
        self.mock_helper.create_connector.return_value = None
        result = self.gcp_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="VertexAI embedding model",
            project_id=self.project_id,
            model_id=self.model_id,
            access_token=self.access_token,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
