# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


class AWSConfig:
    """
    Base class for AWS configuration.
    """

    def __init__(
        self,
        aws_user_name: str,
        aws_role_name: str,
        aws_access_key: str,
        aws_secret_access_key: str,
        aws_session_token: str,
    ):
        """
        Initialize AWS configuration.
        """
        self.aws_user_name = aws_user_name
        self.aws_role_name = aws_role_name
        self.aws_access_key = aws_access_key
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
