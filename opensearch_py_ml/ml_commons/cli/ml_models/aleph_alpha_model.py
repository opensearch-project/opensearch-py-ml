# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ai_connector_helper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class AlephAlphaModel(ModelBase):

    def _get_connector_body(
        self, model_type: str, aleph_alpha_api_key: str
    ) -> Dict[str, Any]:
        """
        Get the connectory body.
        """
        connector_configs = {
            "1": {
                "name": "Aleph Alpha Connector: luminous-base, representation: document",
                "description": "The connector to the Aleph Alpha luminous-base embedding model with representation: document",
                "request_body": '{ "model": "luminous-base", "prompt": "${parameters.input}", "representation": "${parameters.representation}", "normalize": ${parameters.normalize}}',
                "url": "https://api.aleph-alpha.com/semantic_embed",
                "pre_process_function": '\n    StringBuilder builder = new StringBuilder();\n    builder.append("\\"");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append("\\"");\n    def parameters = "{" +"\\"input\\":" + builder + "}";\n    return  "{" +"\\"parameters\\":" + parameters + "}";',
                "post_process_function": '\n      def name = "sentence_embedding";\n      def dataType = "FLOAT32";\n      if (params.embedding == null || params.embedding.length == 0) {\n        return params.message;\n      }\n      def shape = [params.embedding.length];\n      def json = "{" +\n                 "\\"name\\":\\"" + name + "\\"," +\n                 "\\"data_type\\":\\"" + dataType + "\\"," +\n                 "\\"shape\\":" + shape + "," +\n                 "\\"data\\":" + params.embedding +\n                 "}";\n      return json;\n    ',
                "parameters": {"representation": "document"},
            },
            "2": "Custom model",
        }

        # Handle custom model or invalid choice
        if (
            model_type not in connector_configs
            or connector_configs[model_type] == "Custom model"
        ):
            if model_type not in connector_configs:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
                )
            return self.input_custom_model_details()

        config = connector_configs[model_type]

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": "1",
            "protocol": "http",
            "parameters": config["parameters"],
            "credential": {"AlephAlpha_API_Token": aleph_alpha_api_key},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {aleph_alpha_api_key}",
                    },
                    "url": config["url"],
                    "request_body": config["request_body"],
                    **(
                        {"pre_process_function": config["pre_process_function"]}
                        if "pre_process_function" in config
                        else {}
                    ),
                    **(
                        {"post_process_function": config["post_process_function"]}
                        if "post_process_function" in config
                        else {}
                    ),
                }
            ],
        }

    def create_connector(
        self,
        helper: AIConnectorHelper,
        save_config_method: Callable[[str, Dict[str, Any]], None],
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        connector_body: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create Aleph Alpha connector.

        Args:
            helper: Helper instance for OpenSearch connector operations.
            save_config_method: Method to save connector configuration after creation.
            model_name (optional): Specific Aleph Alpha model name.
            api_key (optional): Aleph Alpha API key.
            connector_body (optional): The connector request body.

        Returns:
            bool: True if connector creation successful, False otherwise.
        """
        # Set trusted connector endpoints for Aleph Alpha
        trusted_endpoint = "^https://api\\.aleph-alpha\\.com/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details("Aleph Alpha", self.OPEN_SOURCE, model_name)

        # Prompt for API key
        aleph_alpha_api_key = self.set_api_key(api_key, "Aleph Alpha")

        # Get connector body
        connector_body = connector_body or self._get_connector_body(
            model_type, aleph_alpha_api_key
        )

        # Create connector
        print("\nCreating Aleph Alpha connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=None,
            body=connector_body,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Aleph Alpha connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(connector_id, connector_output)
            return True
        else:
            print(f"{Fore.RED}Failed to create Aleph Alpha connector.{Style.RESET_ALL}")
            return False
