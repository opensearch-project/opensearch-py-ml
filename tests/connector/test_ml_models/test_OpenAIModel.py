# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.connector.ml_models.OpenAIModel import OpenAIModel


class TestOpenAIModel(unittest.TestCase):

    def setUp(self):
        self.mock_iam_role_helper = Mock()
        self.service_type = ""
        self.openai_model = OpenAIModel(self.service_type)

    @patch("builtins.input", side_effect=["", "1", "1"])
    def test_create_openai_connector_open_source_default(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector.return_value = "test-connector-id"

        mock_config = {}
        mock_save_config = Mock()

        self.openai_model.service_type = "open-source"
        self.openai_model.create_openai_connector(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector.assert_called_once()
        mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

    @patch(
        "builtins.input",
        side_effect=[
            "",
            "1",
            "2",
            '{"name": "Custom Model", "description": "Custom description"}',
        ],
    )
    def test_create_openai_connector_open_source_custom(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector.return_value = "test-connector-id"

        mock_config = {}
        mock_save_config = Mock()

        self.openai_model.service_type = "open-source"
        self.openai_model.create_openai_connector(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector.assert_called_once()
        mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

    @patch("builtins.input", side_effect=["", "1", "1"])
    def test_create_openai_connector_managed_default(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_secret.return_value = "test-connector-id"

        mock_config = {}
        mock_save_config = Mock()

        self.openai_model.service_type = "managed"
        self.openai_model.create_openai_connector(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector_with_secret.assert_called_once()
        mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

    @patch(
        "builtins.input",
        side_effect=[
            "",
            "1",
            "2",
            '{"name": "Custom Model", "description": "Custom description"}',
        ],
    )
    def test_create_openai_connector_managed_custom(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_secret.return_value = "test-connector-id"

        mock_config = {}
        mock_save_config = Mock()

        self.openai_model.service_type = "managed"
        self.openai_model.create_openai_connector(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector_with_secret.assert_called_once()
        mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})


if __name__ == "__main__":
    unittest.main()
