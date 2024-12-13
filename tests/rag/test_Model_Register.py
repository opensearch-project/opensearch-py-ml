# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, patch

from opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register import ModelRegister


class TestModelRegister(unittest.TestCase):
    def setUp(self):
        # Sample configuration dictionary
        self.config = {
            "region": "us-east-1",
            "opensearch_username": "admin",
            "opensearch_password": "admin",
            "iam_principal": "arn:aws:iam::123456789012:user/test-user",
            "service_type": "managed",
            "embedding_dimension": "768",
            "opensearch_endpoint": "https://search-domain",
        }
        # Mock OpenSearch client
        self.mock_opensearch_client = MagicMock()
        # OpenSearch domain name
        self.opensearch_domain_name = "test-domain"

        # Correct the patch paths to match the actual module structure
        self.patcher_iam_role_helper = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.IAMRoleHelper"
        )
        self.MockIAMRoleHelper = self.patcher_iam_role_helper.start()

        self.patcher_ai_connector_helper = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.AIConnectorHelper"
        )
        self.MockAIConnectorHelper = self.patcher_ai_connector_helper.start()

        # Patch model classes
        self.patcher_bedrock_model = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.BedrockModel"
        )
        self.MockBedrockModel = self.patcher_bedrock_model.start()

        self.patcher_openai_model = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.OpenAIModel"
        )
        self.MockOpenAIModel = self.patcher_openai_model.start()

        self.patcher_cohere_model = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.CohereModel"
        )
        self.MockCohereModel = self.patcher_cohere_model.start()

        self.patcher_huggingface_model = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.HuggingFaceModel"
        )
        self.MockHuggingFaceModel = self.patcher_huggingface_model.start()

        self.patcher_custom_pytorch_model = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.CustomPyTorchModel"
        )
        self.MockCustomPyTorchModel = self.patcher_custom_pytorch_model.start()

    def tearDown(self):
        self.patcher_iam_role_helper.stop()
        self.patcher_ai_connector_helper.stop()
        self.patcher_bedrock_model.stop()
        self.patcher_openai_model.stop()
        self.patcher_cohere_model.stop()
        self.patcher_huggingface_model.stop()
        self.patcher_custom_pytorch_model.stop()

    @patch("boto3.client")
    def test_initialize_clients_success(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        result = model_register.initialize_clients()
        self.assertTrue(result)
        mock_boto_client.assert_called_with("bedrock-runtime", region_name="us-east-1")

    @patch("boto3.client")
    def test_initialize_clients_failure(self, mock_boto_client):
        mock_boto_client.side_effect = Exception("Client creation failed")
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        result = model_register.initialize_clients()
        self.assertFalse(result)
        mock_boto_client.assert_called_with("bedrock-runtime", region_name="us-east-1")

    @patch("builtins.input", side_effect=["1"])
    def test_prompt_model_registration_register_new_model(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with patch.object(
            model_register, "register_model_interactive"
        ) as mock_register_model_interactive:
            model_register.prompt_model_registration()
            mock_register_model_interactive.assert_called_once()

    @patch("builtins.input", side_effect=["2", "model-id-123"])
    def test_prompt_model_registration_use_existing_model(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with patch.object(model_register, "save_config") as mock_save_config:
            model_register.prompt_model_registration()
            self.assertEqual(
                model_register.config["embedding_model_id"], "model-id-123"
            )
            mock_save_config.assert_called_once_with(model_register.config)

    @patch("builtins.input", side_effect=["invalid"])
    def test_prompt_model_registration_invalid_choice(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with self.assertRaises(SystemExit):
            model_register.prompt_model_registration()

    @patch("builtins.input", side_effect=["1"])
    @patch(
        "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.ModelRegister.initialize_clients",
        return_value=True,
    )
    def test_register_model_interactive_bedrock(
        self, mock_initialize_clients, mock_input
    ):
        self.MockIAMRoleHelper.return_value.get_iam_user_name_from_arn.return_value = (
            "test-user"
        )

        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        model_register.register_model_interactive()

        self.MockBedrockModel.return_value.register_bedrock_model.assert_called_once()

    @patch("builtins.input", side_effect=["2"])
    @patch(
        "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.ModelRegister.initialize_clients",
        return_value=True,
    )
    def test_register_model_interactive_openai_managed(
        self, mock_initialize_clients, mock_input
    ):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        model_register.service_type = "managed"

        self.MockIAMRoleHelper.return_value.get_iam_user_name_from_arn.return_value = (
            "test-user"
        )

        model_register.register_model_interactive()

        self.MockOpenAIModel.return_value.register_openai_model.assert_called_once()

    @patch("builtins.input", side_effect=["2"])
    @patch(
        "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.ModelRegister.initialize_clients",
        return_value=True,
    )
    def test_register_model_interactive_openai_opensource(
        self, mock_initialize_clients, mock_input
    ):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        model_register.service_type = "open-source"

        self.MockIAMRoleHelper.return_value.get_iam_user_name_from_arn.return_value = (
            "test-user"
        )

        model_register.register_model_interactive()

        self.MockOpenAIModel.return_value.register_openai_model_opensource.assert_called_once()

    @patch("builtins.input", side_effect=["invalid"])
    @patch(
        "opensearch_py_ml.ml_commons.rag_pipeline.rag.model_register.ModelRegister.initialize_clients",
        return_value=True,
    )
    def test_register_model_interactive_invalid_choice(
        self, mock_initialize_clients, mock_input
    ):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )

        with patch("builtins.print") as mock_print:
            model_register.register_model_interactive()
            mock_print.assert_called_with(
                "\x1b[31mInvalid choice. Exiting model registration.\x1b[0m"
            )

    @patch("builtins.input", side_effect=["1"])
    def test_prompt_opensource_model_registration_register_now(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with patch.object(
            model_register, "register_model_opensource_interactive"
        ) as mock_register:
            model_register.prompt_opensource_model_registration()
            mock_register.assert_called_once()

    @patch("builtins.input", side_effect=["2"])
    def test_prompt_opensource_model_registration_register_later(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with patch("builtins.print") as mock_print:
            model_register.prompt_opensource_model_registration()
            mock_print.assert_called_with(
                "Skipping model registration. You can register models later using the appropriate commands."
            )

    @patch("builtins.input", side_effect=["invalid"])
    def test_prompt_opensource_model_registration_invalid_choice(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        with patch("builtins.print") as mock_print:
            model_register.prompt_opensource_model_registration()
            mock_print.assert_called_with(
                "\x1b[31mInvalid choice. Skipping model registration.\x1b[0m"
            )

    @patch("builtins.input", side_effect=["3"])
    def test_register_model_opensource_interactive_huggingface(self, mock_input):
        model_register = ModelRegister(
            self.config, self.mock_opensearch_client, self.opensearch_domain_name
        )
        model_register.service_type = "open-source"

        model_register.register_model_opensource_interactive()

        self.MockHuggingFaceModel.return_value.register_huggingface_model.assert_called_once_with(
            model_register.opensearch_client,
            model_register.config,
            model_register.save_config,
        )

    @patch("builtins.input", side_effect=["1"])
    def test_register_model_opensource_interactive_no_opensearch_client(
        self, mock_input
    ):
        model_register = ModelRegister(self.config, None, self.opensearch_domain_name)
        with patch("builtins.print") as mock_print:
            model_register.register_model_opensource_interactive()
            mock_print.assert_called_with(
                "\x1b[31mOpenSearch client is not initialized. Please run setup again.\x1b[0m"
            )


if __name__ == "__main__":
    unittest.main()
