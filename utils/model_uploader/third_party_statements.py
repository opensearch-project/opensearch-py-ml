# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import re

import requests


def generate_thirdpart_statements_for_huggingface_MIT_models(
    model_id: str, mit_license_url: str
) -> str:
    """
    Generate statements text for huggingface MIT-licensed third party model. The result should be put in the final artifact.

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param mit_license_url: the url of the model's MIT license
    :type mit_license_url: string
    :rtype: str
    """

    r = requests.get(mit_license_url)
    assert r.status_code == 200, "Failed to add license file to the model zip file"
    license_text = r.content.decode("utf-8")

    # find the copyright statements from origin MIT license. It looks like: Copyright (c) {year} {authorname}
    copyright_statements = re.findall("Copyright.*\n", license_text)[0].strip()
    huggingface_url = "https://huggingface.co/" + model_id

    full_statements = f"** {model_id}; version  -- {huggingface_url}\n{copyright_statements}\n\n{license_text}"

    return full_statements
