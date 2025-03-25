# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.AlephAlphaModel import (
    AlephAlphaModel,
)


class TestAlephAlphaModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.aleph_alpha_model = AlephAlphaModel()

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value="test_api_key",
    )
    def test_create_aleph_alpha_connector_luminous_base(self, mock_get_password):
        # Test for Luminous-Base embedding model
        result = self.aleph_alpha_model.create_aleph_alpha_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Luminous-Base embedding model",
            api_key="test_api_key",
        )

        # Verify the settings were set
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct payload
        self.mock_helper.create_connector.assert_called_once()
        call_args = self.mock_helper.create_connector.call_args[1]
        self.assertIsNone(call_args["create_connector_role_name"])
        self.assertEqual(
            call_args["payload"]["name"],
            "Aleph Alpha Connector: luminous-base, representation: document",
        )

        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value="test_api_key",
    )
    def test_create_aleph_alpha_connector_custom_model(self, mock_get_password):
        # Test for Custom model
        custom_payload = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

        result = self.aleph_alpha_model.create_aleph_alpha_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Custom model",
            api_key="test_api_key",
            connector_payload=custom_payload,
        )

        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    def test_create_aleph_alpha_connector_failure(self):
        # Test connector creation failure
        self.mock_helper.create_connector.return_value = None

        result = self.aleph_alpha_model.create_aleph_alpha_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Luminous-Base embedding model",
            api_key="test_api_key",
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
