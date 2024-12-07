# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json

import boto3
import requests
from botocore.exceptions import ClientError


class IAMRoleHelper:
    """
    Helper class for managing IAM roles and their interactions with OpenSearch.
    """

    def __init__(
        self,
        region,
        opensearch_domain_url=None,
        opensearch_domain_username=None,
        opensearch_domain_password=None,
        aws_user_name=None,
        aws_role_name=None,
        opensearch_domain_arn=None,
    ):
        """
        Initialize the IAMRoleHelper with AWS and OpenSearch configurations.

        :param region: AWS region.
        :param opensearch_domain_url: URL of the OpenSearch domain.
        :param opensearch_domain_username: Username for OpenSearch domain authentication.
        :param opensearch_domain_password: Password for OpenSearch domain authentication.
        :param aws_user_name: AWS IAM user name.
        :param aws_role_name: AWS IAM role name.
        :param opensearch_domain_arn: ARN of the OpenSearch domain.
        """
        self.region = region
        self.opensearch_domain_url = opensearch_domain_url
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_password = opensearch_domain_password
        self.aws_user_name = aws_user_name
        self.aws_role_name = aws_role_name
        self.opensearch_domain_arn = opensearch_domain_arn

    def role_exists(self, role_name):
        """
        Check if an IAM role exists.

        :param role_name: Name of the IAM role.
        :return: True if the role exists, False otherwise.
        """
        iam_client = boto3.client("iam")

        try:
            iam_client.get_role(RoleName=role_name)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                return False
            else:
                print(f"An error occurred: {e}")
                return False

    def delete_role(self, role_name):
        """
        Delete an IAM role along with its attached policies.

        :param role_name: Name of the IAM role to delete.
        """
        iam_client = boto3.client("iam")

        try:
            # Detach managed policies from the role
            policies = iam_client.list_attached_role_policies(RoleName=role_name)[
                "AttachedPolicies"
            ]
            for policy in policies:
                iam_client.detach_role_policy(
                    RoleName=role_name, PolicyArn=policy["PolicyArn"]
                )
            print(f"All managed policies detached from role {role_name}.")

            # Delete inline policies associated with the role
            inline_policies = iam_client.list_role_policies(RoleName=role_name)[
                "PolicyNames"
            ]
            for policy_name in inline_policies:
                iam_client.delete_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
            print(f"All inline policies deleted from role {role_name}.")

            # Finally, delete the IAM role
            iam_client.delete_role(RoleName=role_name)
            print(f"Role {role_name} deleted.")

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                print(f"Role {role_name} does not exist.")
            else:
                print(f"An error occurred: {e}")

    def create_iam_role(self, role_name, trust_policy_json, inline_policy_json):
        """
        Create a new IAM role with specified trust and inline policies.

        :param role_name: Name of the IAM role to create.
        :param trust_policy_json: Trust policy document in JSON format.
        :param inline_policy_json: Inline policy document in JSON format.
        :return: ARN of the created role or None if creation failed.
        """
        iam_client = boto3.client("iam")

        try:
            # Create the role with the provided trust policy
            create_role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy_json),
                Description="Role with custom trust and inline policies",
            )

            # Retrieve the ARN of the newly created role
            role_arn = create_role_response["Role"]["Arn"]

            # Attach the inline policy to the role
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="InlinePolicy",  # Replace with preferred policy name if needed
                PolicyDocument=json.dumps(inline_policy_json),
            )

            print(f"Created role: {role_name}")
            return role_arn

        except ClientError as e:
            print(f"Error creating the role: {e}")
            return None

    def get_role_arn(self, role_name):
        """
        Retrieve the ARN of an IAM role.

        :param role_name: Name of the IAM role.
        :return: ARN of the role or None if not found.
        """
        if not role_name:
            return None
        iam_client = boto3.client("iam")
        try:
            response = iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                print(f"The requested role {role_name} does not exist")
                return None
            else:
                print(f"An error occurred: {e}")
                return None

    def get_role_details(self, role_name):
        """
        Print detailed information about an IAM role.

        :param role_name: Name of the IAM role.
        """
        iam = boto3.client("iam")

        try:
            response = iam.get_role(RoleName=role_name)
            role = response["Role"]

            print(f"Role Name: {role['RoleName']}")
            print(f"Role ID: {role['RoleId']}")
            print(f"ARN: {role['Arn']}")
            print(f"Creation Date: {role['CreateDate']}")
            print("Assume Role Policy Document:")
            print(
                json.dumps(role["AssumeRolePolicyDocument"], indent=4, sort_keys=True)
            )

            # List and print all inline policies attached to the role
            list_role_policies_response = iam.list_role_policies(RoleName=role_name)

            for policy_name in list_role_policies_response["PolicyNames"]:
                get_role_policy_response = iam.get_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
                print(f"Role Policy Name: {get_role_policy_response['PolicyName']}")
                print("Role Policy Document:")
                print(
                    json.dumps(
                        get_role_policy_response["PolicyDocument"],
                        indent=4,
                        sort_keys=True,
                    )
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                print(f"Role {role_name} does not exist.")
            else:
                print(f"An error occurred: {e}")

    def get_user_arn(self, username):
        """
        Retrieve the ARN of an IAM user.

        :param username: Name of the IAM user.
        :return: ARN of the user or None if not found.
        """
        if not username:
            return None
        iam_client = boto3.client("iam")

        try:
            response = iam_client.get_user(UserName=username)
            user_arn = response["User"]["Arn"]
            return user_arn
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                print(f"IAM user '{username}' not found.")
                return None
            else:
                print(f"An error occurred: {e}")
                return None

    def assume_role(self, role_arn, role_session_name="your_session_name"):
        """
        Assume an IAM role and obtain temporary security credentials.

        :param role_arn: ARN of the IAM role to assume.
        :param role_session_name: Identifier for the assumed role session.
        :return: Temporary security credentials or None if the operation fails.
        """
        sts_client = boto3.client("sts")

        try:
            assumed_role_object = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=role_session_name,
            )

            # Extract temporary credentials from the assumed role
            temp_credentials = assumed_role_object["Credentials"]

            return temp_credentials
        except ClientError as e:
            print(f"Error assuming role: {e}")
            return None

    def map_iam_role_to_backend_role(self, iam_role_arn):
        """
        Map an IAM role to an OpenSearch backend role for access control.

        :param iam_role_arn: ARN of the IAM role to map.
        """
        os_security_role = (
            "ml_full_access"  # Defines the OpenSearch security role to map to
        )
        url = f"{self.opensearch_domain_url}/_plugins/_security/api/rolesmapping/{os_security_role}"

        payload = {"backend_roles": [iam_role_arn]}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.put(
                url,
                auth=(self.opensearch_domain_username, self.opensearch_domain_password),
                json=payload,
                headers=headers,
                verify=True,
            )

            if response.status_code == 200:
                print(
                    f"Successfully mapped IAM role to OpenSearch role '{os_security_role}'."
                )
            else:
                print(
                    f"Failed to map IAM role to OpenSearch role '{os_security_role}'. Status code: {response.status_code}"
                )
                print(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")

    def get_iam_user_name_from_arn(self, iam_principal_arn):
        """
        Extract the IAM user name from an IAM principal ARN.

        :param iam_principal_arn: ARN of the IAM principal.
        :return: IAM user name or None if extraction fails.
        """
        # IAM user ARN format: arn:aws:iam::123456789012:user/user-name
        if iam_principal_arn and ":user/" in iam_principal_arn:
            return iam_principal_arn.split(":user/")[-1]
        else:
            return None
