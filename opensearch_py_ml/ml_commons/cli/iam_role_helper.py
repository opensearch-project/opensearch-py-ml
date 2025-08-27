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
from typing import Any, Dict, Optional, Union

import boto3
import requests
from botocore.exceptions import ClientError
from requests.auth import HTTPBasicAuth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)

# Configure the logger for this module
logger = logging.getLogger(__name__)


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

    def _handle_client_error(self, error, resource_name, resource_type="Role") -> bool:
        """
        Handle ClientError exception.
        """
        if error.response["Error"]["Code"].lower() == self.NO_SUCH_ENTITY:
            logger.warning(f"{resource_type} '{resource_name}' does not exist.")
        else:
            logger.error(f"An error occurred: {error}")
        return False

    def role_exists(self, role_name: str) -> bool:
        """
        Check if an IAM role exists.

        Args:
            role_name: Name of the IAM role.

        Returns:
            bool: True if the role exists, False otherwise.
        """
        try:
            self.iam_client.get_role(RoleName=role_name)
            return True
        except ClientError as e:
            return self._handle_client_error(e, role_name)

    def delete_role(self, role_name: str) -> None:
        """
        Delete an IAM role along with its attached policies.

        Args:
            role_name: Name of the IAM role to delete.
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
            logger.info(f"All managed policies detached from role '{role_name}'.")

            # Delete inline policies associated with the role
            inline_policies = self.iam_client.list_role_policies(RoleName=role_name)[
                "PolicyNames"
            ]
            for policy_name in inline_policies:
                self.iam_client.delete_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
            logger.info(f"All inline policies deleted from role '{role_name}'.")

            # Finally, delete the IAM role
            self.iam_client.delete_role(RoleName=role_name)
            logger.info(f"Role '{role_name}' deleted.")

        except ClientError as e:
            return self._handle_client_error(e, role_name)

    def create_iam_role(
        self,
        role_name: str,
        trust_policy_json: Dict[str, Any],
        inline_policy_json: Dict[str, Any],
        policy_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a new IAM role with specified trust and inline policies.

        Args:
            role_name: Name of the IAM role to create.
            trust_policy_json: Trust policy document in JSON format.
            inline_policy_json: Inline policy document in JSON format.
            policy_name (optional): Name for the inline policy. If not provided, a unique one will be generated.

        Returns:
            Optional[str]:
                - str: ARN of created role if successful
                - None: If role creation fails
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

            logger.info(f"Created role: {role_name} with inline policy: {policy_name}")
            return role_arn

        except ClientError as e:
            logger.error(f"Error creating the role: {e}")
            return None

    def get_role_info(
        self, role_name: str, include_details: bool = False
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Retrieve information about an IAM role.

        Args:
            role_name: Name of the IAM role.
            include_details (optional): If False, returns only the role's ARN. If True, returns a dictionary with full role details.

        Returns:
            Optional[Union[str, Dict[str, Any]]]:
                - str: Role ARN if include_details=False
                - Dict: Complete role details if include_details=True
                - None: If role not found or error occurs
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
            return self._handle_client_error(e, role_name)

    def get_role_arn(self, role_name: str) -> Optional[str]:
        """
        Retrieve the ARN of an IAM role.

        Args:
            role_name: Name of the IAM role.

        Returns:
            Optional[str]:
                - str: Role ARN if found.
                - None: If role doesn't exist or error occurs.
        """
        if not role_name:
            return None
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            # Return ARN of the role
            return response["Role"]["Arn"]
        except ClientError as e:
            return self._handle_client_error(e, role_name)

    def get_user_arn(self, username: str) -> Optional[str]:
        """
        Retrieve the ARN of an IAM user.

        Args:
            username: Name of the IAM user.

        Returns:
            Optional[str]:
                - str: User's ARN if found
                - None: If username is empty or user not found
        """
        if not username:
            return None
        try:
            response = self.iam_client.get_user(UserName=username)
            return response["User"]["Arn"]
        except ClientError as e:
            return self._handle_client_error(e, username, "User")

    def map_iam_role_to_backend_role(
        self, role_arn: str, os_security_role: str = "ml_full_access"
    ) -> None:
        """
        Maps an IAM role to an OpenSearch security backend role.

        Args:
            role_arn: ARN of the IAM role to be mapped.
            os_security_role (optional): The OpenSearch security role name.
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
            logger.info(response.text)
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
            logger.info(response.text)

    def assume_role(
        self,
        role_arn: str,
        role_session_name: Optional[str] = None,
        session: Optional[boto3.Session] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Assume an IAM role and obtain temporary security credentials.

        Args:
            role_arn: ARN of the IAM role to assume.
            role_session_name (optional): Identifier for the assumed role session.
            session (optional): Optional boto3 session object. Defaults to the class-level sts_client.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with temporary security credentials and metadata, or None on failure.
        """
        if not role_arn:
            logger.error("Role ARN is required.")
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

            logger.info(
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
            logger.error(f"Error assuming role {role_arn}: {error_code} - {e}")
            return None

    def get_iam_user_name_from_arn(self, iam_principal_arn: str) -> Optional[str]:
        """
        Extract the IAM user name from an IAM principal ARN.

        Args:
            iam_principal_arn: ARN of the IAM principal. Expected format: arn:aws:iam::<account-id>:user/<user-name>

        Returns:
            Optional[str]:
                - str: Extracted user name if successful
                - None: If ARN is invalid, empty, or extraction fails
        """
        try:
            if (
                iam_principal_arn
                and iam_principal_arn.startswith("arn:aws:iam::")
                and ":user/" in iam_principal_arn
            ):
                return iam_principal_arn.split(":user/")[-1]
        except Exception as e:
            logger.error(f"Error extracting IAM user name: {e}")
        return None
