# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json


class ModelBase:

    def get_custom_model_details(self, default_input):
        print(
            "\nDo you want to use the default configuration or provide custom model settings?"
        )
        print("1. Use default configuration")
        print("2. Provide custom model settings")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            return default_input
        elif choice == "2":
            print("Please enter your model details as a JSON object.")
            print("Example:")
            print(json.dumps(default_input, indent=2))
            json_input = input("Enter your JSON object: ").strip()
            try:
                custom_details = json.loads(json_input)
                return custom_details
            except json.JSONDecodeError as e:
                print(f"Invalid JSON input: {e}")
                return None
        else:
            print("Invalid choice. Aborting connector creation.")
            return None
