# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
import uuid
from datetime import datetime

import boto3
import requests
from botocore.exceptions import ClientError
from requests.auth import HTTPBasicAuth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)


class IAMRoleHelper:
    """
    Helper class for managing IAM roles and their interactions with OpenSearch.
    """

    NO_SUCH_ENTITY = "nosuchentity"
    NOT_FOUND = "not_found"

    def __init__(
        self,
        opensearch_config: OpenSearchDomainConfig,
        aws_config: AWSConfig,
    ):
        """
        Initialize the IAMRoleHelper with AWS and OpenSearch configurations.
        """
        self.opensearch_config = opensearch_config
        self.aws_config = aws_config

        self.iam_client = boto3.client(
            "iam",
            aws_access_key_id=self.aws_config.aws_access_key,
            aws_secret_access_key=self.aws_config.aws_secret_access_key,
            aws_session_token=self.aws_config.aws_session_token,
        )
        self.sts_client = boto3.client(
            "sts",
            region_name=self.opensearch_config.opensearch_domain_region,
            aws_access_key_id=self.aws_config.aws_access_key,
            aws_secret_access_key=self.aws_config.aws_secret_access_key,
            aws_session_token=self.aws_config.aws_session_token,
        )

    def role_exists(self, role_name):
        """
        Check if an IAM role exists.
        :param role_name: Name of the IAM role.
        :return: True if the role exists, False otherwise.
        """
        try:
            self.iam_client.get_role(RoleName=role_name)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
                print(f"The requested role '{role_name}' does not exist.")
            else:
                print(f"An error occurred: {e}")
            return False

    def delete_role(self, role_name):
        """
        Delete an IAM role along with its attached policies.
        :param role_name: Name of the IAM role to delete.
        """
        try:
            # Detach any managed policies from the role
            policies = self.iam_client.list_attached_role_policies(RoleName=role_name)[
                "AttachedPolicies"
            ]
            for policy in policies:
                self.iam_client.detach_role_policy(
                    RoleName=role_name, PolicyArn=policy["PolicyArn"]
                )
            print(f"All managed policies detached from role '{role_name}'.")

            # Delete inline policies associated with the role
            inline_policies = self.iam_client.list_role_policies(RoleName=role_name)[
                "PolicyNames"
            ]
            for policy_name in inline_policies:
                self.iam_client.delete_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
            print(f"All inline policies deleted from role '{role_name}'.")

            # Finally, delete the IAM role
            self.iam_client.delete_role(RoleName=role_name)
            print(f"Role '{role_name}' deleted.")

        except ClientError as e:
            if e.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
                print(f"Role '{role_name}' does not exist.")
            else:
                print(f"An error occurred: {e}")

    def create_iam_role(
        self,
        role_name,
        trust_policy_json,
        inline_policy_json,
        policy_name=None,
    ):
        """
        Create a new IAM role with specified trust and inline policies.
        :param role_name: Name of the IAM role to create.
        :param trust_policy_json: Trust policy document in JSON format.
        :param inline_policy_json: Inline policy document in JSON format.
        :param policy_name: Optional. If not provided, a unique one will be generated.
        :return: ARN of the created role or None if creation failed.
        """
        try:
            # Create the role with the provided trust policy
            create_role_response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy_json),
                Description="Role with custom trust and inline policies",
            )

            # Retrieve the ARN of the newly created role
            role_arn = create_role_response["Role"]["Arn"]

            # If policy_name is not provided, generate a unique one
            if not policy_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                policy_name = f"InlinePolicy-{role_name}-{timestamp}"

            # Attach the inline policy to the role
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(inline_policy_json),
            )

            print(f"Created role: {role_name} with inline policy: {policy_name}")
            return role_arn

        except ClientError as e:
            print(f"Error creating the role: {e}")
            return None

    def get_role_info(self, role_name, include_details=False):
        """
        Retrieve information about an IAM role.
        :param role_name: Name of the IAM role.
        :param include_details: If False, returns only the role's ARN.
                               If True, returns a dictionary with full role details.
        :return: ARN or dict of role details. Returns None if not found.
        """
        if not role_name:
            return None

        try:
            response = self.iam_client.get_role(RoleName=role_name)
            role = response["Role"]
            role_arn = role["Arn"]

            if not include_details:
                return role_arn

            # Build a detailed dictionary
            role_details = {
                "RoleName": role["RoleName"],
                "RoleId": role["RoleId"],
                "Arn": role_arn,
                "CreationDate": role["CreateDate"],
                "AssumeRolePolicyDocument": role["AssumeRolePolicyDocument"],
                "InlinePolicies": {},
            }

            # List and retrieve any inline policies
            list_role_policies_response = self.iam_client.list_role_policies(
                RoleName=role_name
            )
            for policy_name in list_role_policies_response["PolicyNames"]:
                get_role_policy_response = self.iam_client.get_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
                role_details["InlinePolicies"][policy_name] = get_role_policy_response[
                    "PolicyDocument"
                ]

            return role_details

        except ClientError as e:
            if e.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
                print(f"Role '{role_name}' does not exist.")
            else:
                print(f"An error occurred: {e}")
            return None

    def get_role_arn(self, role_name):
        """
        Retrieve the ARN of an IAM role.
        :param username: Name of the IAM role.
        :return: ARN of the role or None if not found.
        """
        if not role_name:
            return None
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            # Return ARN of the role
            return response["Role"]["Arn"]
        except ClientError as e:
            if e.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
                print(f"IAM role '{role_name}' not found.")
            else:
                print(f"An error occurred: {e}")
            return None

    def get_user_arn(self, username):
        """
        Retrieve the ARN of an IAM user.
        :param username: Name of the IAM user.
        :return: ARN of the user or None if not found.
        """
        if not username:
            return None
        try:
            response = self.iam_client.get_user(UserName=username)
            return response["User"]["Arn"]
        except ClientError as e:
            if e.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
                print(f"IAM user '{username}' not found.")
            else:
                print(f"An error occurred: {e}")
            return None

    def map_iam_role_to_backend_role(self, role_arn, os_security_role="ml_full_access"):
        """
        Maps an IAM role to an OpenSearch security backend role.
        :param role_arn: ARN of the IAM role to be mapped.
        :param: os_security_role: The OpenSearch security role name.
        :return: None
        """
        url = f"{self.opensearch_config.opensearch_domain_endpoint}/_plugins/_security/api/rolesmapping/{os_security_role}"
        r = requests.get(
            url,
            auth=HTTPBasicAuth(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
        )
        role_mapping = json.loads(r.text)
        headers = {"Content-Type": "application/json"}
        if (
            "status" in role_mapping
            and role_mapping["status"].lower() == self.NOT_FOUND
        ):
            data = {"backend_roles": [role_arn]}
            response = requests.put(
                url,
                headers=headers,
                data=json.dumps(data),
                auth=HTTPBasicAuth(
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                ),
            )
            print(response.text)
        else:
            role_mapping = role_mapping[os_security_role]
            role_mapping["backend_roles"].append(role_arn)
            data = [
                {
                    "op": "replace",
                    "path": "/backend_roles",
                    "value": list(set(role_mapping["backend_roles"])),
                }
            ]
            response = requests.patch(
                url,
                headers=headers,
                data=json.dumps(data),
                auth=HTTPBasicAuth(
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                ),
            )
            print(response.text)

    def assume_role(self, role_arn, role_session_name=None, session=None):
        """
        Assume an IAM role and obtain temporary security credentials.
        :param role_arn: ARN of the IAM role to assume.
        :param role_session_name: Identifier for the assumed role session.
        :param session: Optional boto3 session object. Defaults to the class-level sts_client.
        :return: Dictionary with temporary security credentials and metadata, or None on failure.
        """
        if not role_arn:
            logging.error("Role ARN is required.")
            return None

        sts_client = session.client("sts") if session else self.sts_client

        role_session_name = role_session_name or f"session-{uuid.uuid4()}"

        try:
            assumed_role_object = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=role_session_name,
            )

            temp_credentials = assumed_role_object["Credentials"]
            expiration = temp_credentials["Expiration"]

            logging.info(
                f"Assumed role: {role_arn}. Temporary credentials valid until: {expiration}"
            )

            return {
                "credentials": {
                    "AccessKeyId": temp_credentials["AccessKeyId"],
                    "SecretAccessKey": temp_credentials["SecretAccessKey"],
                    "SessionToken": temp_credentials["SessionToken"],
                },
                "expiration": expiration,
                "session_name": role_session_name,
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logging.error(f"Error assuming role {role_arn}: {error_code} - {e}")
            return None

    def get_iam_user_name_from_arn(self, iam_principal_arn):
        """
        Extract the IAM user name from an IAM principal ARN.
        :param iam_principal_arn: ARN of the IAM principal. Expected format: arn:aws:iam::<account-id>:user/<user-name>
        :return: IAM user name if extraction is successful, None otherwise.
        """
        try:
            if (
                iam_principal_arn
                and iam_principal_arn.startswith("arn:aws:iam::")
                and ":user/" in iam_principal_arn
            ):
                return iam_principal_arn.split(":user/")[-1]
        except Exception as e:
            print(f"Error extracting IAM user name: {e}")
        return None
