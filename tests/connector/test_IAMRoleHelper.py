# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper


class TestIAMRoleHelper(unittest.TestCase):
    def setUp(self):
        """
        Create an IAMRoleHelper instance with mock configurations.
        Patching boto3 clients so that no real AWS calls are made.
        """
        self.opensearch_domain_region = "us-east-1"

        # Patches for the boto3 clients
        self.patcher_iam = patch("boto3.client")
        self.mock_boto_client = self.patcher_iam.start()

        # Mock the IAM client and STS client
        self.mock_iam_client = MagicMock()
        self.mock_sts_client = MagicMock()

        # Configure the mock_boto_client to return the respective mocks
        self.mock_boto_client.side_effect = lambda service_name, region_name=None: {
            "iam": self.mock_iam_client,
            "sts": self.mock_sts_client,
        }[service_name]

        # Instantiate our class under test
        self.helper = IAMRoleHelper(
            opensearch_domain_region=self.opensearch_domain_region
        )

    def tearDown(self):
        self.patcher_iam.stop()

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


if __name__ == "__main__":
    unittest.main()
