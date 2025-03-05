# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class AlephAlphaModel(ModelBase):

    def create_aleph_alpha_connector(self, helper, config, save_config_method):
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
        api_key = input("Enter your Aleph Alpha API key: ").strip()

        create_connector_role_name = "create_aleph_alpha_connector_role"

        # Default connector input
        default_connector_input = {
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

        # Get model details from user
        create_connector_input = self.get_custom_model_details(default_connector_input)
        if not create_connector_input:
            return  # Abort if no valid input

        # Create connector
        print("Creating Aleph Alpha connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=create_connector_role_name,
            payload=create_connector_input,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Aleph Alpha connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            # Update config with connector ID if needed
            config["connector_id"] = connector_id
            save_config_method(config)
            return True
        else:
            print(f"{Fore.RED}Failed to create Aleph Alpha connector.{Style.RESET_ALL}")
            return False
