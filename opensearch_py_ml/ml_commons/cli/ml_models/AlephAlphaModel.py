# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class AlephAlphaModel(ModelBase):
    def create_aleph_alpha_connector(
        self,
        helper,
        save_config_method,
        model_name=None,
        api_key=None,
        connector_payload=None,
    ):
        """
        Create Aleph Alpha connector.
        """
        # Set trusted connector endpoints for Aleph Alpha
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.aleph-alpha\\.com/.*$"
                ]
            }
        }
        helper.opensearch_client.cluster.put_settings(body=settings_body)

        # Prompt for necessary input
        if model_name == "Luminous-Base embedding model":
            model_type = "1"
        elif model_name == "Custom model":
            model_type = "2"
        else:
            print("\nPlease select a model for the connector creation: ")
            print("1. Luminous-Base embedding model")
            print("2. Custom model")
            model_type = input("Enter your choice (1-2): ").strip()

        setup = Setup()
        if not api_key:
            api_key = setup.get_password_with_asterisks(
                "Enter your Aleph Alpha API key: "
            )

        if model_type == "1":
            connector_payload = {
                "name": "Aleph Alpha Connector: luminous-base, representation: document",
                "description": "The connector to the Aleph Alpha luminous-base embedding model with representation: document",
                "version": "1",
                "protocol": "http",
                "parameters": {"representation": "document"},
                "credential": {"AlephAlpha_API_Token": api_key},
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://api.aleph-alpha.com/semantic_embed",
                        "headers": {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        "request_body": '{ "model": "luminous-base", "prompt": "${parameters.input}", "representation": "${parameters.representation}", "normalize": ${parameters.normalize}}',
                        "pre_process_function": '\n    StringBuilder builder = new StringBuilder();\n    builder.append("\\"");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append("\\"");\n    def parameters = "{" +"\\"input\\":" + builder + "}";\n    return  "{" +"\\"parameters\\":" + parameters + "}";',
                        "post_process_function": '\n      def name = "sentence_embedding";\n      def dataType = "FLOAT32";\n      if (params.embedding == null || params.embedding.length == 0) {\n        return params.message;\n      }\n      def shape = [params.embedding.length];\n      def json = "{" +\n                 "\\"name\\":\\"" + name + "\\"," +\n                 "\\"data_type\\":\\"" + dataType + "\\"," +\n                 "\\"shape\\":" + shape + "," +\n                 "\\"data\\":" + params.embedding +\n                 "}";\n      return json;\n    ',
                    }
                ],
            }
        elif model_type == "2":
            if not connector_payload:
                connector_payload = self.input_custom_model_details()
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
            )
            if not connector_payload:
                connector_payload = self.input_custom_model_details()

        # Create connector
        print("Creating Aleph Alpha connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=None,
            payload=connector_payload,
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
