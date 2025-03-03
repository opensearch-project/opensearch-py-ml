# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, patch

from opensearch_py_ml.ml_commons.connector.connector_create import Create


class TestCreate(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.create_instance = Create()
        self.mock_config = {
            "opensearch_domain_region": "us-west-2",
            "opensearch_domain_name": "test-domain",
            "opensearch_domain_username": "admin",
            "opensearch_domain_password": "password",
            "aws_user_name": "test-user-arn",
            "aws_role_name": "test-role-arn",
            "opensearch_domain_endpoint": "https://test-domain.amazonaws.com",
            "connector_blueprint": {
                "connector_role_inline_policy": {"Statement": []},
                "connector_role_name": "test-connector-role",
                "create_connector_role_name": "test-create-role",
                "connector_config": {"name": "test-connector"},
                "secret_name": "test-secret",
                "secret_value": "test-value",
            },
        }

    @patch("opensearch_py_ml.ml_commons.connector.connector_create.AIConnectorHelper")
    def test_create_command_managed_success(self, mock_ai_helper):
        """Test successful connector creation in managed mode."""
        # Setup Create instance
        self.create_instance.service_type = "managed"
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.initialize_opensearch_client = MagicMock(return_value=True)
        self.create_instance.save_config = MagicMock()

        # Mock AIConnectorHelper instance
        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.create_connector_with_role.return_value = (
            "test-connector-id"
        )

        # Execute
        result = self.create_instance.create_command()

        # Assert
        self.assertTrue(result)
        mock_helper_instance.create_connector_with_role.assert_called_once_with(
            connector_role_inline_policy=self.mock_config["connector_blueprint"][
                "connector_role_inline_policy"
            ],
            connector_role_name=self.mock_config["connector_blueprint"][
                "connector_role_name"
            ],
            create_connector_role_name=self.mock_config["connector_blueprint"][
                "create_connector_role_name"
            ],
            create_connector_input=self.mock_config["connector_blueprint"][
                "connector_config"
            ],
        )
        self.create_instance.save_config.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.connector.connector_create.AIConnectorHelper")
    def test_create_command_opensource_success(self, mock_ai_helper):
        """Test successful connector creation in open-source mode."""
        # Setup
        self.create_instance.service_type = "open-source"
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.initialize_opensearch_client = MagicMock(return_value=True)
        self.create_instance.save_config = MagicMock()

        # Mock AIConnectorHelper instance
        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.create_connector_with_secret.return_value = (
            "test-connector-id"
        )

        # Execute
        result = self.create_instance.create_command()

        # Assert
        self.assertTrue(result)
        mock_helper_instance.create_connector_with_secret.assert_called_once_with(
            secret_name=self.mock_config["connector_blueprint"]["secret_name"],
            secret_value=self.mock_config["connector_blueprint"]["secret_value"],
            connector_role_name=self.mock_config["connector_blueprint"][
                "connector_role_name"
            ],
            create_connector_role_name=self.mock_config["connector_blueprint"][
                "create_connector_role_name"
            ],
            create_connector_input=self.mock_config["connector_blueprint"][
                "connector_config"
            ],
        )

    def test_create_command_no_config(self):
        """Test create command with no configuration."""
        self.create_instance.load_config = MagicMock(return_value=None)
        result = self.create_instance.create_command()
        self.assertFalse(result)

    def test_create_command_no_blueprint(self):
        """Test create command with no blueprint in config."""
        config_without_blueprint = self.mock_config.copy()
        del config_without_blueprint["connector_blueprint"]
        self.create_instance.load_config = MagicMock(
            return_value=config_without_blueprint
        )
        self.create_instance.initialize_opensearch_client = MagicMock(return_value=True)
        result = self.create_instance.create_command()
        self.assertFalse(result)

    def test_create_command_client_initialization_failure(self):
        """Test create command when client initialization fails."""
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.initialize_opensearch_client = MagicMock(
            return_value=False
        )
        result = self.create_instance.create_command()
        self.assertFalse(result)

    @patch("opensearch_py_ml.ml_commons.connector.AIConnectorHelper")
    def test_create_command_connector_creation_failure(self, mock_ai_helper):
        """Test create command when connector creation fails."""
        # Setup
        self.create_instance.service_type = "managed"
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.initialize_opensearch_client = MagicMock(return_value=True)

        # Mock AIConnectorHelper instance to return None (failure)
        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.create_connector_with_role.return_value = None

        # Execute
        result = self.create_instance.create_command()

        # Assert
        self.assertFalse(result)

    def test_create_command_exception_handling(self):
        """Test create command exception handling."""
        self.create_instance.load_config = MagicMock(
            side_effect=Exception("Test error")
        )
        result = self.create_instance.create_command()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
