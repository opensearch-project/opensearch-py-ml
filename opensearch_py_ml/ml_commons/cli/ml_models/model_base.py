# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

from rich.console import Console

# Initialize Rich console for enhanced CLI outputs
console = Console()
from colorama import Fore, Style


class ModelBase:
    def input_custom_model_details(self, external=False):
        if external:
            print(
                f"{Fore.YELLOW}\nIMPORTANT: When customizing the connector configuration, ensure you include the following in the 'headers' section:"
            )
            print(f'{Fore.YELLOW}{Style.BRIGHT}"Authorization": "${{auth}}"')
            print(
                f"{Fore.YELLOW}This placeholder will be automatically replaced with the secure reference to your API key.\n"
            )
        print("Please enter your model details as a JSON object.")
        print("\nClick the link below for examples of the connector blueprint: ")
        console.print("[bold]Amazon OpenSearch Service:[/bold]")
        print(
            "https://github.com/opensearch-project/ml-commons/tree/2.x/docs/tutorials/aws"
        )
        console.print("\n[bold]Open-Source Service:[/bold]")
        print(
            "https://github.com/opensearch-project/ml-commons/tree/2.x/docs/remote_inference_blueprints"
        )
        print("\nEnter your JSON object (press Enter twice when done): ")
        json_input = ""
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)

        json_input = "\n".join(lines)

        try:
            custom_details = json.loads(json_input)
            return custom_details
        except json.JSONDecodeError as e:
            print(f"Invalid JSON input: {e}")
            return None
