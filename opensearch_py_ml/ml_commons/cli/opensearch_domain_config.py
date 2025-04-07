# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


class OpenSearchDomainConfig:
    """
    Base class for OpenSearch domain configuration.
    """

    def __init__(
        self,
        opensearch_domain_region: str,
        opensearch_domain_name: str,
        opensearch_domain_username: str,
        opensearch_domain_password: str,
        opensearch_domain_endpoint: str,
    ):
        """
        Initialize OpenSearch domain configuration.
        """
        self.opensearch_domain_region = opensearch_domain_region
        self.opensearch_domain_name = opensearch_domain_name
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_password = opensearch_domain_password
        self.opensearch_domain_endpoint = opensearch_domain_endpoint
