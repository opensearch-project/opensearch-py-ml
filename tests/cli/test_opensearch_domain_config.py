# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest

from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)


class TestOpenSearchDomainConfig(unittest.TestCase):
    def test_opensearch_config_initialization(self):
        """Test OpenSearchDomainConfig initialization with valid parameters"""
        # Test data
        test_config = {
            "opensearch_domain_region": "test-region",
            "opensearch_domain_name": "test-domain-name",
            "opensearch_domain_username": "admin",
            "opensearch_domain_password": "password",
            "opensearch_domain_endpoint": "test-domain-endpoint",
        }

        # Create instance
        opensearch_config = OpenSearchDomainConfig(**test_config)

        # Verify all attributes are set correctly
        self.assertEqual(
            opensearch_config.opensearch_domain_region,
            test_config["opensearch_domain_region"],
        )
        self.assertEqual(
            opensearch_config.opensearch_domain_name,
            test_config["opensearch_domain_name"],
        )
        self.assertEqual(
            opensearch_config.opensearch_domain_username,
            test_config["opensearch_domain_username"],
        )
        self.assertEqual(
            opensearch_config.opensearch_domain_password,
            test_config["opensearch_domain_password"],
        )
        self.assertEqual(
            opensearch_config.opensearch_domain_endpoint,
            test_config["opensearch_domain_endpoint"],
        )


if __name__ == "__main__":
    unittest.main()
