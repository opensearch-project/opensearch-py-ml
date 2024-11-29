# CustomPyTorchModel.py
import json
from colorama import Fore, Style
import os
import time

class CustomPyTorchModel:
    def __init__(self, aws_region, opensearch_domain_name, opensearch_username, opensearch_password, iam_role_helper):
        """
        Initializes the CustomPyTorchModel with necessary configurations.
        
        Args:
            aws_region (str): AWS region.
            opensearch_domain_name (str): OpenSearch domain name.
            opensearch_username (str): OpenSearch username.
            opensearch_password (str): OpenSearch password.
            iam_role_helper (IAMRoleHelper): Instance of IAMRoleHelper.
        """
        self.aws_region = aws_region
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password
        self.iam_role_helper = iam_role_helper

    def register_custom_pytorch_model(self, opensearch_client, config, save_config_method):
        """
        Register a custom PyTorch embedding model in open-source OpenSearch.

        Args:
            opensearch_client: OpenSearch client instance.
            config (dict): Configuration dictionary.
            save_config_method (function): Method to save the configuration.
        """
        print("\nDo you want to use the default configuration or provide custom settings?")
        print("1. Use default configuration")
        print("2. Provide custom settings")
        config_choice = input("Enter your choice (1-2): ").strip()

        if config_choice == '1':
            # Use default configurations
            model_path = input("Enter the path to your PyTorch model file (.pt or .pth): ").strip()
            if not os.path.isfile(model_path):
                print(f"{Fore.RED}Model file not found at '{model_path}'. Aborting.{Style.RESET_ALL}")
                return

            model_name = os.path.basename(model_path).split('.')[0]
            model_payload = {
                "name": f"custom_pytorch_{model_name}",
                "model_format": "TORCH_SCRIPT",
                "model_config": {
                    "embedding_dimension": config.get('embedding_dimension', 768),
                    "framework_type": "CUSTOM",
                    "model_type": "bert"
                },
                "description": f"Custom PyTorch model: {model_name}"
            }
        elif config_choice == '2':
            # Get custom configurations
            model_path = input("Enter the path to your PyTorch model file (.pt or .pth): ").strip()
            if not os.path.isfile(model_path):
                print(f"{Fore.RED}Model file not found at '{model_path}'. Aborting.{Style.RESET_ALL}")
                return

            print("\nPlease enter your model details as a JSON object.")
            print("Example:")
            example_payload = {
                "name": "custom_pytorch_model",
                "model_format": "TORCH_SCRIPT",
                "model_config": {
                    "embedding_dimension": config.get('embedding_dimension', 768),
                    "framework_type": "CUSTOM",
                    "model_type": "bert"
                },
                "description": "Custom PyTorch model for semantic search"
            }
            print(json.dumps(example_payload, indent=2))
            
            model_payload = self.get_custom_json_input()
            if not model_payload:
                return
        else:
            print(f"{Fore.RED}Invalid choice. Aborting model registration.{Style.RESET_ALL}")
            return

        # Upload the model file to OpenSearch
        try:
            with open(model_path, 'rb') as f:
                model_content = f.read()

            # Use the ML plugin's model upload API
            upload_response = opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_upload",
                params={"model_name": model_payload['name']},
                body=model_content,
                headers={'Content-Type': 'application/octet-stream'}
            )
            if 'model_id' not in upload_response:
                print(f"{Fore.RED}Failed to upload model. Response: {upload_response}{Style.RESET_ALL}")
                return
            model_id = upload_response['model_id']
            print(f"{Fore.GREEN}Model uploaded successfully. Model ID: {model_id}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error uploading model: {ex}{Style.RESET_ALL}")
            return

        # Add the model_id to the payload
        model_payload['model_id'] = model_id

        # Register the model
        try:
            response = opensearch_client.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=model_payload
            )
            task_id = response.get('task_id')
            if task_id:
                print(f"{Fore.GREEN}Model registration initiated. Task ID: {task_id}{Style.RESET_ALL}")
                # Wait for the task to complete and retrieve the model_id
                registered_model_id = self.wait_for_model_registration(opensearch_client, task_id)
                if registered_model_id:
                    # Deploy the model
                    deploy_response = opensearch_client.transport.perform_request(
                        method="POST",
                        url=f"/_plugins/_ml/models/{registered_model_id}/_deploy"
                    )
                    print(f"{Fore.GREEN}Model deployed successfully. Model ID: {registered_model_id}{Style.RESET_ALL}")
                    config['embedding_model_id'] = registered_model_id
                    save_config_method(config)
                else:
                    print(f"{Fore.RED}Model registration failed or timed out.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to initiate model registration. Response: {response}{Style.RESET_ALL}")
        except Exception as ex:
            print(f"{Fore.RED}Error registering model: {ex}{Style.RESET_ALL}")

    def get_custom_json_input(self):
        """Helper method to get custom JSON input from the user."""
        json_input = input("Enter your JSON object: ").strip()
        try:
            return json.loads(json_input)
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Invalid JSON input: {e}{Style.RESET_ALL}")
            return None

    def wait_for_model_registration(self, opensearch_client, task_id, timeout=600, interval=10):
        """
        Wait for the model registration task to complete and return the model_id.

        Args:
            opensearch_client: OpenSearch client instance.
            task_id (str): Task ID to monitor.
            timeout (int): Maximum time to wait in seconds.
            interval (int): Time between status checks in seconds.

        Returns:
            str or None: The model ID if successful, else None.
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = opensearch_client.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                state = response.get('state')
                if state == 'COMPLETED':
                    model_id = response.get('model_id')
                    return model_id
                elif state in ['FAILED', 'STOPPED']:
                    print(f"{Fore.RED}Model registration task {task_id} failed with state: {state}{Style.RESET_ALL}")
                    return None
                else:
                    print(f"Model registration task {task_id} is in state: {state}. Waiting...")
                    time.sleep(interval)
            except Exception as ex:
                print(f"{Fore.RED}Error checking task status: {ex}{Style.RESET_ALL}")
                time.sleep(interval)
        print(f"{Fore.RED}Timed out waiting for model registration to complete.{Style.RESET_ALL}")
        return None