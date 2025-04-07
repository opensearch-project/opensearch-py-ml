# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import sys
import unittest
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from requests.auth import HTTPBasicAuth

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)
from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper


class TestIAMRoleHelper(unittest.TestCase):
    def setUp(self):
        """
        Create an IAMRoleHelper instance with mock configurations.
        Patching boto3 clients so that no real AWS calls are made.
        """
        # Create OpenSearchDomainConfig
        self.opensearch_config = OpenSearchDomainConfig(
            opensearch_domain_region="us-east-1",
            opensearch_domain_name="test-domain",
            opensearch_domain_username="admin",
            opensearch_domain_password="password",
            opensearch_domain_endpoint="test-domain-url",
        )
        # Create AWSConfig
        self.aws_config = AWSConfig(
            aws_user_name="",
            aws_role_name="",
            aws_access_key="test-access-key",
            aws_secret_access_key="test-secret-access-key",
            aws_session_token="test-session-token",
        )

        # Patches for the boto3 clients
        self.patcher_boto3 = patch("boto3.client")
        self.mock_boto_client = self.patcher_boto3.start()

        # Mock the IAM client and STS client
        self.mock_iam_client = Mock()
        self.mock_sts_client = Mock()

        # Configure the mock_boto_client to return the respective mocks
        def mock_client(service_name, **kwargs):
            if service_name == "iam":
                return self.mock_iam_client
            elif service_name == "sts":
                return self.mock_sts_client
            raise ValueError(f"Unexpected service: {service_name}")

        self.mock_boto_client.side_effect = mock_client

        # Instantiate our class under test
        self.helper = IAMRoleHelper(
            opensearch_config=self.opensearch_config, aws_config=self.aws_config
        )

    def tearDown(self):
        self.patcher_boto3.stop()

    def test_role_exists_found(self):
        """Test role_exists returns True when role is found."""
        # Mock successful get_role call
        self.mock_iam_client.get_role.return_value = {
            "Role": {"RoleName": "my-test-role"}
        }

        result = self.helper.role_exists("my-test-role")
        self.assertTrue(result)
        self.mock_iam_client.get_role.assert_called_once_with(RoleName="my-test-role")

    def test_role_exists_not_found(self):
        """Test role_exists returns False when role is not found."""
        # Mock get_role call to raise NoSuchEntity
        error_response = {
            "Error": {"Code": "NoSuchEntity", "Message": "Role not found"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        result = self.helper.role_exists("non-existent-role")
        self.assertFalse(result)

    def test_role_exists_other_error(self):
        """Test role_exists returns False (and prints error) when another ClientError occurs."""
        error_response = {
            "Error": {"Code": "SomeOtherError", "Message": "Unexpected error"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        result = self.helper.role_exists("some-role")
        self.assertFalse(result)

    def test_delete_role_happy_path(self):
        """Test delete_role successfully detaches and deletes a role."""
        # Mock listing attached policies
        self.mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {"PolicyArn": "arn:aws:iam::123456789012:policy/testPolicy"}
            ]
        }
        # Mock listing inline policies
        self.mock_iam_client.list_role_policies.return_value = {
            "PolicyNames": ["InlinePolicyTest"]
        }

        self.helper.delete_role("my-test-role")

        # Verify detach calls
        self.mock_iam_client.detach_role_policy.assert_called_once_with(
            RoleName="my-test-role",
            PolicyArn="arn:aws:iam::123456789012:policy/testPolicy",
        )
        # Verify delete inline policy call
        self.mock_iam_client.delete_role_policy.assert_called_once_with(
            RoleName="my-test-role", PolicyName="InlinePolicyTest"
        )
        # Verify delete_role call
        self.mock_iam_client.delete_role.assert_called_once_with(
            RoleName="my-test-role"
        )

    def test_delete_role_no_such_entity(self):
        """Test delete_role prints message if role does not exist."""
        error_response = {
            "Error": {"Code": "NoSuchEntity", "Message": "Role not found"}
        }
        self.mock_iam_client.list_attached_role_policies.side_effect = ClientError(
            error_response, "ListAttachedRolePolicies"
        )

        self.helper.delete_role("non-existent-role")
        # We expect it to print a message, and not raise. The method should handle it.

    def test_delete_role_other_error(self):
        """Test delete_role prints error for unexpected ClientError."""
        error_response = {
            "Error": {"Code": "SomeOtherError", "Message": "Unexpected error"}
        }
        self.mock_iam_client.list_attached_role_policies.side_effect = ClientError(
            error_response, "ListAttachedRolePolicies"
        )

        self.helper.delete_role("some-role")

    def test_create_iam_role_happy_path(self):
        """Test create_iam_role creates a role with trust and inline policy."""
        # Mock create_role response
        self.mock_iam_client.create_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/my-test-role",
                "RoleName": "my-test-role",
            }
        }
        trust_policy = {"Version": "2012-10-17", "Statement": []}
        inline_policy = {"Version": "2012-10-17", "Statement": []}

        role_arn = self.helper.create_iam_role(
            role_name="my-test-role",
            trust_policy_json=trust_policy,
            inline_policy_json=inline_policy,
            policy_name="myInlinePolicy",
        )

        # Verify calls
        self.mock_iam_client.create_role.assert_called_once()
        self.mock_iam_client.put_role_policy.assert_called_once()
        self.assertEqual(role_arn, "arn:aws:iam::123456789012:role/my-test-role")

    def test_create_iam_role_failure(self):
        """Test create_iam_role returns None if creation fails."""
        error_response = {
            "Error": {"Code": "SomeOtherError", "Message": "Role creation failure"}
        }
        self.mock_iam_client.create_role.side_effect = ClientError(
            error_response, "CreateRole"
        )

        trust_policy = {"Version": "2012-10-17", "Statement": []}
        inline_policy = {"Version": "2012-10-17", "Statement": []}

        role_arn = self.helper.create_iam_role(
            role_name="my-test-role",
            trust_policy_json=trust_policy,
            inline_policy_json=inline_policy,
        )

        self.assertIsNone(role_arn)
        self.mock_iam_client.put_role_policy.assert_not_called()

    def test_create_iam_role_no_policy_name(self):
        """Test create_iam_role if no policy name provided."""
        # Mock create_role response
        self.mock_iam_client.create_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/my-test-role",
                "RoleName": "my-test-role",
            }
        }
        role_name = "my-test-role"
        trust_policy = {"Version": "2012-10-17", "Statement": []}
        inline_policy = {"Version": "2012-10-17", "Statement": []}

        role_arn = self.helper.create_iam_role(
            role_name="my-test-role",
            trust_policy_json=trust_policy,
            inline_policy_json=inline_policy,
        )
        self.mock_iam_client.put_role_policy.assert_called_once()

        actual_args = self.mock_iam_client.put_role_policy.call_args[1]
        self.assertEqual(actual_args["RoleName"], role_name)
        self.assertTrue(
            actual_args["PolicyName"].startswith(f"InlinePolicy-{role_name}-")
        )
        self.assertEqual(actual_args["PolicyDocument"], json.dumps(inline_policy))
        self.assertEqual(role_arn, "arn:aws:iam::123456789012:role/my-test-role")

    def test_get_role_info_arn_only(self):
        """Test get_role_info returns role ARN only when include_details=False."""
        self.mock_iam_client.get_role.return_value = {
            "Role": {
                "RoleName": "my-test-role",
                "Arn": "arn:aws:iam::123456789012:role/my-test-role",
            }
        }

        arn = self.helper.get_role_info("my-test-role", include_details=False)
        self.assertEqual(arn, "arn:aws:iam::123456789012:role/my-test-role")

    def test_get_role_info_details(self):
        """Test get_role_info returns detailed info when include_details=True."""
        # Mock get_role
        self.mock_iam_client.get_role.return_value = {
            "Role": {
                "RoleName": "my-test-role",
                "RoleId": "AIDA12345EXAMPLE",
                "Arn": "arn:aws:iam::123456789012:role/my-test-role",
                "CreateDate": datetime(2020, 1, 1),
                "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": []},
            }
        }
        # Mock list_role_policies
        self.mock_iam_client.list_role_policies.return_value = {
            "PolicyNames": ["inlinePolicyTest"]
        }
        # Mock get_role_policy
        self.mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {"Version": "2012-10-17", "Statement": []}
        }

        details = self.helper.get_role_info("my-test-role", include_details=True)
        self.assertIsInstance(details, dict)
        self.assertEqual(details["RoleName"], "my-test-role")
        self.assertEqual(details["InlinePolicies"].keys(), {"inlinePolicyTest"})

    def test_get_role_info_not_found(self):
        """Test get_role_info returns None if role is not found."""
        error_response = {
            "Error": {"Code": "NoSuchEntity", "Message": "Role not found"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        details = self.helper.get_role_info("non-existent-role", include_details=True)
        self.assertIsNone(details)

    def test_get_role_info_no_role_name(self):
        """Test get_role_info returns None if role_name not provided."""
        details = self.helper.get_role_info("", include_details=True)
        self.assertIsNone(details)

    def test_get_role_info_client_error(self):
        """Test get_role_info when a ClientError occurs."""
        error_response = {
            "Error": {"Code": "SomeError", "Message": "Something went wrong"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        # Capture stdout to verify print output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = self.helper.get_role_info("test-role")
            self.assertIsNone(result)
            expected_error = "An error occurred: An error occurred (SomeError) when calling the GetRole operation: Something went wrong"
            self.assertIn(expected_error, captured_output.getvalue().strip())
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    def test_get_role_arn_success(self):
        """Test get_role_arn returns ARN if role is found."""
        self.mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:user/TestRole",
                "RoleName": "TestRole",
            }
        }

        arn = self.helper.get_role_arn("TestRole")
        self.assertEqual(arn, "arn:aws:iam::123456789012:user/TestRole")

    def test_get_role_arn_not_found(self):
        """Test get_role_arn returns None if user does not exist."""
        error_response = {
            "Error": {"Code": "NoSuchEntity", "Message": "Role not found"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        arn = self.helper.get_role_arn("NonExistentUser")
        self.assertIsNone(arn)

    def test_get_role_arn_no_rolename(self):
        """Test get_role_arn returns None if rolename not provided."""
        arn = self.helper.get_role_arn("")
        self.assertIsNone(arn)

    def test_get_role_arn_client_error(self):
        """Test get_role_arn when a ClientError occurs."""
        error_response = {
            "Error": {"Code": "SomeError", "Message": "Something went wrong"}
        }
        self.mock_iam_client.get_role.side_effect = ClientError(
            error_response, "GetRole"
        )

        # Capture stdout to verify print output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = self.helper.get_role_arn("test-role")
            self.assertIsNone(result)
            expected_error = "An error occurred: An error occurred (SomeError) when calling the GetRole operation: Something went wrong"
            self.assertIn(expected_error, captured_output.getvalue().strip())
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    def test_get_user_arn_success(self):
        """Test get_user_arn returns ARN if user is found."""
        self.mock_iam_client.get_user.return_value = {
            "User": {
                "Arn": "arn:aws:iam::123456789012:user/TestUser",
                "UserName": "TestUser",
            }
        }

        arn = self.helper.get_user_arn("TestUser")
        self.assertEqual(arn, "arn:aws:iam::123456789012:user/TestUser")

    def test_get_user_arn_not_found(self):
        """Test get_user_arn returns None if user does not exist."""
        error_response = {
            "Error": {"Code": "NoSuchEntity", "Message": "User not found"}
        }
        self.mock_iam_client.get_user.side_effect = ClientError(
            error_response, "GetUser"
        )

        arn = self.helper.get_user_arn("NonExistentUser")
        self.assertIsNone(arn)

    def test_get_user_arn_no_username(self):
        """Test get_user_arn returns None if username not provided."""
        arn = self.helper.get_user_arn("")
        self.assertIsNone(arn)

    def test_get_user_arn_client_error(self):
        """Test get_user_arn when a ClientError occurs."""
        error_response = {
            "Error": {"Code": "SomeError", "Message": "Something went wrong"}
        }
        self.mock_iam_client.get_user.side_effect = ClientError(
            error_response, "GetUser"
        )

        # Capture stdout to verify print output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = self.helper.get_user_arn("test-user")
            self.assertIsNone(result)
            expected_error = "An error occurred: An error occurred (SomeError) when calling the GetUser operation: Something went wrong"
            self.assertIn(expected_error, captured_output.getvalue().strip())
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    @patch("requests.get")
    @patch("requests.put")
    def test_map_iam_role_to_backend_role_not_found(self, mock_put, mock_get):
        """Test map_iam_role_to_backend_role when role doesn't exist (NOT_FOUND case)."""
        # Setup
        role_arn = "arn:aws:iam::123456789012:role/test-role"
        os_security_role = "ml_full_access"

        # Mock GET response for non-existent role
        mock_get_response = Mock()
        mock_get_response.text = json.dumps({"status": "not_found"})
        mock_get.return_value = mock_get_response

        # Mock PUT response
        mock_put_response = Mock()
        mock_put_response.text = "Role mapped successfully"
        mock_put.return_value = mock_put_response

        # Execute
        self.helper.map_iam_role_to_backend_role(role_arn, os_security_role)

        # Verify GET request
        expected_url = f"{self.helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_security/api/rolesmapping/{os_security_role}"
        mock_get.assert_called_once_with(
            expected_url,
            auth=HTTPBasicAuth(
                self.helper.opensearch_config.opensearch_domain_username,
                self.helper.opensearch_config.opensearch_domain_password,
            ),
        )

        # Verify PUT request
        expected_data = {"backend_roles": [role_arn]}
        mock_put.assert_called_once_with(
            expected_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(expected_data),
            auth=HTTPBasicAuth(
                self.helper.opensearch_config.opensearch_domain_username,
                self.helper.opensearch_config.opensearch_domain_password,
            ),
        )

    @patch("requests.get")
    @patch("requests.patch")
    def test_map_iam_role_to_backend_role_exists(self, mock_patch, mock_get):
        """Test map_iam_role_to_backend_role when role already exists."""
        # Setup
        role_arn = "arn:aws:iam::123456789012:role/test-role"
        existing_role_arn = "arn:aws:iam::123456789012:role/existing-role"
        os_security_role = "ml_full_access"

        # Mock GET response for existing role
        mock_get_response = Mock()
        mock_get_response.text = json.dumps(
            {"ml_full_access": {"backend_roles": [existing_role_arn]}}
        )
        mock_get.return_value = mock_get_response

        # Mock PATCH response
        mock_patch_response = Mock()
        mock_patch_response.text = "Role updated successfully"
        mock_patch.return_value = mock_patch_response

        # Execute
        self.helper.map_iam_role_to_backend_role(role_arn, os_security_role)

        # Verify PATCH request
        expected_url = f"{self.helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_security/api/rolesmapping/{os_security_role}"
        expected_data = [
            {
                "op": "replace",
                "path": "/backend_roles",
                "value": list(set([existing_role_arn, role_arn])),
            }
        ]
        mock_patch.assert_called_once_with(
            expected_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(expected_data),
            auth=HTTPBasicAuth(
                self.helper.opensearch_config.opensearch_domain_username,
                self.helper.opensearch_config.opensearch_domain_password,
            ),
        )

    def test_assume_role_happy_path(self):
        """Test assume_role returns credentials on success."""
        # Mock assume_role response
        mock_credentials = {
            "AccessKeyId": "AKIAEXAMPLE",
            "SecretAccessKey": "SECRET",
            "SessionToken": "TOKEN",
            "Expiration": datetime.utcnow() + timedelta(hours=1),
        }
        self.mock_sts_client.assume_role.return_value = {
            "Credentials": mock_credentials
        }

        role_arn = "arn:aws:iam::123456789012:role/my-test-role"
        response = self.helper.assume_role(role_arn, "test-session")
        self.assertIsNotNone(response)
        self.assertIn("credentials", response)
        self.assertEqual(response["credentials"]["AccessKeyId"], "AKIAEXAMPLE")

    def test_assume_role_no_arn(self):
        """Test assume_role returns None if no ARN is provided."""
        response = self.helper.assume_role(None, "test-session")
        self.assertIsNone(response)
        self.mock_sts_client.assume_role.assert_not_called()

    def test_assume_role_failure(self):
        """Test assume_role returns None if STS call fails."""
        error_response = {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Not authorized to assume role",
            }
        }
        self.mock_sts_client.assume_role.side_effect = ClientError(
            error_response, "AssumeRole"
        )

        role_arn = "arn:aws:iam::123456789012:role/my-test-role"
        response = self.helper.assume_role(role_arn, "test-session")
        self.assertIsNone(response)

    def test_get_iam_user_name_from_arn_valid(self):
        """Test get_iam_user_name_from_arn returns the username part of the ARN."""
        arn = "arn:aws:iam::123456789012:user/MyUser"
        username = self.helper.get_iam_user_name_from_arn(arn)
        self.assertEqual(username, "MyUser")

    def test_get_iam_user_name_from_arn_invalid_format(self):
        """Test get_iam_user_name_from_arn returns None for invalid format."""
        arn = "arn:aws:iam::123456789012:role/MyRole"
        username = self.helper.get_iam_user_name_from_arn(arn)
        self.assertIsNone(username)

    def test_get_iam_user_name_from_arn_none_input(self):
        """Test get_iam_user_name_from_arn returns None if input is None."""
        username = self.helper.get_iam_user_name_from_arn(None)
        self.assertIsNone(username)

    @patch("builtins.print")
    def test_get_iam_user_name_from_arn_exception(self, mock_print):
        """Test get_iam_user_name_from_arn for exception handling and print output."""
        result = self.helper.get_iam_user_name_from_arn(123)

        self.assertIsNone(result)
        expected_error = (
            "Error extracting IAM user name: 'int' object has no attribute 'startswith'"
        )
        mock_print.assert_called_once_with(expected_error)


if __name__ == "__main__":
    unittest.main()
