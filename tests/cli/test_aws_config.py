# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig


class TestAWSConfig(unittest.TestCase):
    def test_aws_config_initialization(self):
        """Test AWSConfig initialization with valid parameters"""
        # Test data
        test_config = {
            "aws_user_name": "test-user",
            "aws_role_name": "test-role",
            "aws_access_key": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
            "aws_session_token": "test-session-token",
        }

        # Create instance
        aws_config = AWSConfig(**test_config)

        # Verify all attributes are set correctly
        self.assertEqual(aws_config.aws_user_name, test_config["aws_user_name"])
        self.assertEqual(aws_config.aws_role_name, test_config["aws_role_name"])
        self.assertEqual(aws_config.aws_access_key, test_config["aws_access_key"])
        self.assertEqual(
            aws_config.aws_secret_access_key, test_config["aws_secret_access_key"]
        )
        self.assertEqual(aws_config.aws_session_token, test_config["aws_session_token"])


if __name__ == "__main__":
    unittest.main()
