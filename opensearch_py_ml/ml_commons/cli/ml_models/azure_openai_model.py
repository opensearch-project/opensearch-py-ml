# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class AzureOpenAIModel(ModelBase):

    def _get_connector_body(
        self, model_type, resource_name, deployment_name, api_version, openai_api_key
    ):
        """
        Get the connectory body
        """
        connector_configs = {
            "1": {
                "name": "Azure OpenAI chat completion connector",
                "description": "Connector for Azure OpenAI chat completion model",
                "model": "gpt-4",
                "request_body": '{ "messages": ${parameters.messages}, "temperature": ${parameters.temperature} }',
                "url": "https://${parameters.endpoint}/openai/deployments/${parameters.deploy-name}/chat/completions?api-version=${parameters.api-version}",
                "parameters": {"temperature": 0.7},
            },
            "2": {
                "name": "Azure OpenAI embedding connector",
                "description": "Connector for Azure OpenAI embedding model",
                "model": "text-embedding-ada-002",
                "request_body": '{ "input": ${parameters.input}}',
                "url": "https://${parameters.endpoint}/openai/deployments/${parameters.deploy-name}/embeddings?api-version=${parameters.api-version}",
                "pre_process_function": "connector.pre_process.openai.embedding",
                "post_process_function": "connector.post_process.openai.embedding",
            },
            "3": "Custom model",
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

        # Base parameters that all connectors need
        base_parameters = {
            "endpoint": f"{resource_name}.openai.azure.com/",
            "deploy-name": f"{deployment_name}",
            "api-version": f"{api_version}",
            "model": config["model"],
        }

        # Merge with model-specific parameters if any
        parameters = {**base_parameters, **config.get("parameters", {})}

        # Return the connector body
        return {
            "name": config["name"],
            "description": config["description"],
            "version": "1.0",
            "protocol": "http",
            "parameters": parameters,
            "credential": {"openAI_key": openai_api_key},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"api-key": openai_api_key},
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
        helper,
        save_config_method,
        model_name=None,
        api_key=None,
        resource_name=None,
        deployment_name=None,
        api_version=None,
        connector_body=None,
    ):
        """
        Create Azure OpenAI connector.
        """
        # Set trusted connector endpoints for Azure OpenAI
        trusted_endpoint = "^https://.*\\.openai\\.azure\\.com/.*$"
        self.set_trusted_endpoint(helper, trusted_endpoint)

        # Prompt to choose model
        model_type = self.get_model_details(
            "Azure OpenAI", self.OPEN_SOURCE, model_name
        )

        # Prompt for API key
        openai_api_key = self.set_api_key(api_key, "OpenAI")

        # Prompt for resource and deployment name, API version
        resource_name = (
            resource_name or input("Enter your Azure OpenAI resource name: ").strip()
        )
        deployment_name = (
            deployment_name
            or input("Enter your Azure OpenAI deployment name: ").strip()
        )
        api_version = (
            api_version or input("Enter your Azure OpenAI API version: ").strip()
        )

        # Get connector body
        connector_body = connector_body or self._get_connector_body(
            model_type, resource_name, deployment_name, api_version, openai_api_key
        )

        # Create connector
        print("\nCreating Azure OpenAI connector...")
        connector_id = helper.create_connector(
            create_connector_role_name=None,
            body=connector_body,
        )

        if connector_id:
            print(
                f"{Fore.GREEN}\nSuccessfully created Azure OpenAI connector with ID: {connector_id}{Style.RESET_ALL}"
            )
            connector_output = helper.get_connector(connector_id)
            save_config_method(connector_id, connector_output)
            return True
        else:
            print(
                f"{Fore.RED}Failed to create Azure OpenAI connector.{Style.RESET_ALL}"
            )
            return False
