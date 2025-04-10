# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from io import StringIO
from typing import List
from unittest.mock import MagicMock, call, patch

from opensearch_py_ml.ml_commons.cli.connector_list import ConnectorInfo, ModelInfo
from opensearch_py_ml.ml_commons.cli.connector_manager import ConnectorManager


class TestConnectorManager(unittest.TestCase):
    def setUp(self):
        self.connector_manager = ConnectorManager()
        self.test_opensource_connectors: List[ConnectorInfo] = [
            ConnectorInfo(
                id=1,
                name="Aleph Alpha",
                file_name="aleph_alpha_model",
                connector_class="AlephAlphaModel",
                init_params=[],
                connector_params=["model_name", "api_key", "connector_body"],
                available_models=[
                    ModelInfo(id="1", name="Luminous-Base embedding model"),
                    ModelInfo(id="2", name="Custom model"),
                ],
            ),
            ConnectorInfo(
                id=2,
                name="Bedrock",
                file_name="bedrock_model",
                connector_class="BedrockModel",
                init_params=[],
                connector_params=["model_name", "region", "connector_body"],
                available_models=[
                    ModelInfo(id="1", name="Titan embedding model"),
                ],
            ),
        ]
        self.test_managed_connectors: List[ConnectorInfo] = [
            ConnectorInfo(
                id=1,
                name="Managed Connector",
                file_name="managed_connector",
                connector_class="ManagedModel",
                init_params=[],
                connector_params=["model_name", "endpoint"],
                available_models=[
                    ModelInfo(id="1", name="Managed Model 1"),
                ],
            )
        ]
        self.test_connector_info = ConnectorInfo(
            id=1,
            name="Test Connector",
            file_name="test_connector",
            connector_class="TestModel",
            init_params=["param1", "param2"],
            connector_params=["param1", "param2", "param3"],
            available_models=[ModelInfo(id="1", name="Test Model 1")],
        )
        self.test_config = {"service_type": "open-source", "some_config": "value"}
        self.test_connector_config = {
            "setup_config_path": "test/setup/path",
            "connector_name": "Test Connector",
            "model_name": "test_model",
            "api_key": "test_key",
        }

        # Set up connector lists
        self.connector_manager.connector_list._opensource_connectors = (
            self.test_opensource_connectors
        )
        self.connector_manager.connector_list._managed_connectors = (
            self.test_managed_connectors
        )

    def test_get_connectors_opensource(self):
        """Test get_connectors with open-source service type"""
        connectors = self.connector_manager.get_connectors("open-source")
        self.assertEqual(connectors, self.test_opensource_connectors)
        self.assertEqual(len(connectors), 2)
        self.assertEqual(connectors[0].name, "Aleph Alpha")
        self.assertEqual(connectors[1].name, "Bedrock")

    def test_get_connectors_managed(self):
        """Test get_connectors with amazon-opensearch-service service type"""
        connectors = self.connector_manager.get_connectors("amazon-opensearch-service")
        self.assertEqual(connectors, self.test_managed_connectors)
        self.assertEqual(len(connectors), 1)
        self.assertEqual(connectors[0].name, "Managed Connector")

    def test_get_connectors_invalid_service_type(self):
        """Test get_connectors with invalid service type"""
        with self.assertRaises(ValueError) as context:
            self.connector_manager.get_connectors("invalid-service")
        self.assertEqual(
            str(context.exception), "Unknown service type: invalid-service"
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_available_connectors_opensource(self, mock_stdout):
        """Test print_available_connectors with open-source connectors"""
        self.connector_manager.print_available_connectors("open-source")

        expected_output = (
            "\nPlease select a supported connector to create:\n"
            "1. Aleph Alpha\n"
            "2. Bedrock\n"
            "Enter your choice (1-2): "
        )
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_available_connectors_managed(self, mock_stdout):
        """Test print_available_connectors with managed connectors"""
        self.connector_manager.print_available_connectors("amazon-opensearch-service")

        expected_output = (
            "\nPlease select a supported connector to create:\n"
            "1. Managed Connector\n"
            "Enter your choice (1-1): "
        )
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_available_connectors_no_connectors(self, mock_stdout):
        """Test print_available_connectors with empty connector list"""
        # Temporarily set empty connector lists
        self.connector_manager.connector_list._opensource_connectors = []
        self.connector_manager.print_available_connectors("open-source")

        expected_output = "\nNo connectors available for open-source\n"
        self.assertEqual(mock_stdout.getvalue(), expected_output)

        # Restore original connector lists
        self.connector_manager.connector_list._opensource_connectors = (
            self.test_opensource_connectors
        )

    def test_get_connector_by_id_opensource(self):
        """Test get_connector_by_id with open-source connectors"""
        # Test getting first connector
        connector = self.connector_manager.get_connector_by_id(1, "open-source")
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.name, "Aleph Alpha")

        # Test getting second connector
        connector = self.connector_manager.get_connector_by_id(2, "open-source")
        self.assertEqual(connector.id, 2)
        self.assertEqual(connector.name, "Bedrock")

    def test_get_connector_by_id_managed(self):
        """Test get_connector_by_id with managed connectors"""
        connector = self.connector_manager.get_connector_by_id(
            1, "amazon-opensearch-service"
        )
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.name, "Managed Connector")

    def test_get_connector_by_id_invalid_id(self):
        """Test get_connector_by_id with invalid connector ID"""
        with self.assertRaises(ValueError):
            self.connector_manager.get_connector_by_id(999, "open-source")

    def test_get_connector_by_id_invalid_service_type(self):
        """Test get_connector_by_id with invalid service type"""
        with self.assertRaises(ValueError) as context:
            self.connector_manager.get_connector_by_id(1, "invalid-service")
        self.assertEqual(
            str(context.exception), "Unknown service type: invalid-service"
        )

    def test_get_connector_by_name_opensource(self):
        """Test get_connector_by_name with open-source connectors"""
        # Test getting first connector
        connector = self.connector_manager.get_connector_by_name(
            "Aleph Alpha", "open-source"
        )
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.name, "Aleph Alpha")

        # Test getting second connector
        connector = self.connector_manager.get_connector_by_name(
            "Bedrock", "open-source"
        )
        self.assertEqual(connector.id, 2)
        self.assertEqual(connector.name, "Bedrock")

    def test_get_connector_by_name_managed(self):
        """Test get_connector_by_name with managed connectors"""
        connector = self.connector_manager.get_connector_by_name(
            "Managed Connector", "amazon-opensearch-service"
        )
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.name, "Managed Connector")

    def test_get_connector_by_name_invalid_name(self):
        """Test get_connector_by_name with invalid connector name"""
        with self.assertRaises(ValueError):
            self.connector_manager.get_connector_by_name(
                "Invalid Connector", "open-source"
            )

    def test_get_connector_by_name_invalid_service_type(self):
        """Test get_connector_by_name with invalid service type"""
        with self.assertRaises(ValueError) as context:
            self.connector_manager.get_connector_by_name("Bedrock", "invalid-service")
        self.assertEqual(
            str(context.exception), "Unknown service type: invalid-service"
        )

    def test_get_available_models_opensource(self):
        """Test get_available_models for open-source connectors"""
        models = self.connector_manager.get_available_models(
            "Aleph Alpha", "open-source"
        )
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "Luminous-Base embedding model")
        self.assertEqual(models[1].name, "Custom model")

        models = self.connector_manager.get_available_models("Bedrock", "open-source")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Titan embedding model")

    def test_get_available_models_managed(self):
        """Test get_available_models for managed connectors"""
        models = self.connector_manager.get_available_models(
            "Managed Connector", "amazon-opensearch-service"
        )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Managed Model 1")

    def test_get_available_models_nonexistent_connector(self):
        """Test get_available_models with non-existent connector"""
        models = self.connector_manager.get_available_models(
            "NonExistent Connector", "open-source"
        )
        self.assertEqual(models, [])

    @patch("builtins.__import__")
    def test_get_connector_class_success(self, mock_import):
        """Test get_connector_class with valid connector"""
        mock_class = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, "AlephAlphaModel", mock_class)
        mock_import.return_value = mock_module

        result = self.connector_manager.get_connector_class("Aleph Alpha")
        mock_import.assert_called_once_with(
            "opensearch_py_ml.ml_commons.cli.ml_models.aleph_alpha_model",
            fromlist=["AlephAlphaModel"],
        )
        self.assertEqual(result, mock_class)

    @patch("builtins.__import__")
    def test_get_connector_class_nonexistent_connector(self, mock_import):
        """Test get_connector_class with non-existent connector"""
        result = self.connector_manager.get_connector_class("NonExistent")
        self.assertIsNone(result)
        mock_import.assert_not_called()

    def test_get_connector_info_opensource(self):
        """Test get_connector_info with open-source connector"""
        connector = self.connector_manager.get_connector_info("Aleph Alpha")
        self.assertIsNotNone(connector)
        self.assertEqual(connector.name, "Aleph Alpha")
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.connector_class, "AlephAlphaModel")

    def test_get_connector_info_managed(self):
        """Test get_connector_info with managed connector"""
        connector = self.connector_manager.get_connector_info("Managed Connector")
        self.assertIsNotNone(connector)
        self.assertEqual(connector.name, "Managed Connector")
        self.assertEqual(connector.id, 1)
        self.assertEqual(connector.connector_class, "ManagedModel")

    def test_get_connector_info_nonexistent(self):
        """Test get_connector_info with non-existent connector"""
        connector = self.connector_manager.get_connector_info("NonExistent")
        self.assertIsNone(connector)

    def test_create_model_instance_with_opensearch_config(self):
        """Test create_model_instance with opensearch_config parameters"""
        opensearch_config = {
            "param1": "value1",
            "param2": "value2",
            "extra_param": "extra_value",
        }
        config = {}
        mock_class = MagicMock()
        self.connector_manager.create_model_instance(
            self.test_connector_info, mock_class, opensearch_config, config
        )
        mock_class.assert_called_once_with(param1="value1", param2="value2")

    def test_create_model_instance_with_config(self):
        """Test create_model_instance with config parameters"""
        opensearch_config = {}
        config = {"param1": "value1", "param2": "value2", "extra_param": "extra_value"}
        mock_class = MagicMock()
        self.connector_manager.create_model_instance(
            self.test_connector_info, mock_class, opensearch_config, config
        )
        mock_class.assert_called_once_with(param1="value1", param2="value2")

    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper")
    def test_create_connector_instance_with_config_path(self, mock_ai_helper):
        """Test create_connector_instance with connector config path"""
        connector_config_params = {
            "param1": "config_value1",
            "param2": "config_value2",
            "extra_param": "extra_value",
        }
        mock_model = MagicMock()
        result = self.connector_manager.create_connector_instance(
            connector_config_path="test/path",
            connector_config_params=connector_config_params,
            connector_info=self.test_connector_info,
            opensearch_config={},
            config={},
            model=mock_model,
            ai_helper=mock_ai_helper,
        )
        self.assertEqual(result, mock_model.create_connector.return_value)

    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper")
    def test_create_connector_instance_with_opensearch_config(self, mock_ai_helper):
        """Test create_connector_instance with opensearch config"""
        opensearch_config = {"param1": "value1", "param2": None}
        mock_model = MagicMock()
        result = self.connector_manager.create_connector_instance(
            connector_config_path=None,
            connector_config_params={},
            connector_info=self.test_connector_info,
            opensearch_config=opensearch_config,
            config={},
            model=mock_model,
            ai_helper=mock_ai_helper,
        )
        # Check that create_connector was called with correct kwargs
        expected_kwargs = {"param1": "value1"}

        mock_model.create_connector.assert_called_once_with(
            mock_ai_helper, self.connector_manager.connector_output, **expected_kwargs
        )
        self.assertEqual(result, mock_model.create_connector.return_value)

    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper")
    def test_create_connector_instance_with_config(self, mock_ai_helper):
        """Test create_connector_instance with general config"""
        config = {"param1": None, "param2": "value2"}
        mock_model = MagicMock()
        result = self.connector_manager.create_connector_instance(
            connector_config_path=None,
            connector_config_params={},
            connector_info=self.test_connector_info,
            opensearch_config={},
            config=config,
            model=mock_model,
            ai_helper=mock_ai_helper,
        )
        # Check that create_connector was called with correct kwargs
        expected_kwargs = {"param2": "value2"}

        mock_model.create_connector.assert_called_once_with(
            mock_ai_helper, self.connector_manager.connector_output, **expected_kwargs
        )
        self.assertEqual(result, mock_model.create_connector.return_value)

    @patch("builtins.print")
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.load_config"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.load_and_check_config"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.get_connector_by_name"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.get_connector_class"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.create_model_instance"
    )
    def test_initialize_create_connector_with_config_path(
        self,
        mock_create_model,
        mock_get_class,
        mock_get_connector,
        mock_load_check_config,
        mock_load_config,
        mock_print,
    ):
        """Test initialize_create_connector with config path"""
        # Setup mocks
        mock_load_config.return_value = self.test_config
        mock_get_connector.return_value = self.test_connector_info
        mock_get_class.return_value = MagicMock()
        mock_create_model.return_value = MagicMock()
        mock_load_check_config.return_value = (
            "ai_helper",
            self.test_config,
            "service_type",
            self.test_connector_config,
        )

        # Call method
        self.connector_manager.initialize_create_connector("test/config/path")

        # Verify calls
        mock_load_config.assert_any_call("test/config/path", "connector")
        mock_load_check_config.assert_called_once()
        mock_get_connector.assert_called_once()
        mock_get_class.assert_called_once()
        mock_create_model.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.load_config"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager._check_config"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.print_available_connectors"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.get_connector_by_id"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.get_connector_class"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.connector_manager.ConnectorManager.create_model_instance"
    )
    def test_initialize_create_connector_without_config_path(
        self,
        mock_create_model,
        mock_get_class,
        mock_get_connector_id,
        mock_print_connectors,
        mock_check_config,
        mock_load_config,
        mock_print,
        mock_input,
    ):
        """Test initialize_create_connector without config path"""
        # Setup mocks
        mock_input.side_effect = ["test/setup/path", "1"]
        mock_load_config.return_value = self.test_config
        mock_check_config.return_value = MagicMock()
        mock_get_connector_id.return_value = self.test_connector_info
        mock_get_class.return_value = MagicMock()
        mock_create_model.return_value = MagicMock()

        # Call method
        self.connector_manager.initialize_create_connector()

        # Verify calls
        mock_input.assert_has_calls(
            [
                call("\nEnter the path to your existing setup configuration file: "),
                call().strip(),
            ]
        )
        mock_load_config.assert_called_once_with("test/setup/path")
        mock_check_config.assert_called_once()
        mock_print_connectors.assert_called_once_with("open-source")
        mock_get_connector_id.assert_called_once_with(1, "open-source")
        mock_get_class.assert_called_once()
        mock_create_model.assert_called_once()

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_initialize_create_connector_no_connector_config(
        self, mock_input, mock_print
    ):
        """Test initialize_create_connector with no connector configuration found"""
        self.connector_manager.load_config = MagicMock(return_value={})
        result = self.connector_manager.initialize_create_connector(
            "test-connector-config.yml"
        )
        self.assertFalse(result)

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_initialize_create_connector_no_setup_config(self, mock_input, mock_print):
        """Test initialize_create_connector with no setup configuration found"""
        self.connector_manager.load_config = MagicMock(return_value=None)
        result = self.connector_manager.initialize_create_connector()
        self.assertFalse(result)

    @patch("builtins.input", side_effect=[""])
    def test_initialize_create_connector_fail_configuration_check(self, mock_input):
        """Test initialize_create_connector when configuration validity check fails"""
        self.connector_manager.load_config = MagicMock(return_value=self.test_config)
        self.connector_manager.check_config = MagicMock(return_value=False)
        result = self.connector_manager.initialize_create_connector()
        self.assertFalse(result)

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_initialize_create_connector_invalid_connector_name(
        self, mock_input, mock_print
    ):
        """Test initialize_create_connector when the connector name is invalid and ValueError is raised"""
        self.connector_manager.load_and_check_config = MagicMock(
            return_value=(
                "ai_helper",
                self.test_config,
                "service_type",
                self.test_connector_config,
            )
        )
        self.connector_manager.load_config = MagicMock(return_value=self.test_config)
        self.connector_manager.get_connector_by_name = MagicMock(side_effect=ValueError)
        result = self.connector_manager.initialize_create_connector(
            "test-connector-config.yml"
        )
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn(
            "Invalid connector choice. Operation cancelled.",
            mock_print.call_args[0][0],
        )

    @patch("builtins.print")
    @patch("builtins.input", return_value="0")
    def test_initialize_create_connector_invalid_connector_id(
        self, mock_input, mock_print
    ):
        """Test initialize_create_connector when the connector ID is invalid and ValueError is raised"""
        self.connector_manager.load_and_check_config = MagicMock(
            return_value=(
                "ai_helper",
                self.test_config,
                "service_type",
                self.test_connector_config,
            )
        )
        self.connector_manager.print_available_connectors = MagicMock()
        self.connector_manager.get_connector_by_id = MagicMock(side_effect=ValueError)
        result = self.connector_manager.initialize_create_connector()
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn(
            "Invalid connector choice. Operation cancelled.",
            mock_print.call_args[0][0],
        )

    @patch("builtins.input", side_effect=[""])
    def test_initialize_create_connector_exception_handling(self, mock_input):
        """Test initialize_create_connector for exception handling"""
        self.connector_manager.load_config = MagicMock(
            side_effect=Exception("Test error")
        )
        result = self.connector_manager.initialize_create_connector()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
