# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
from string import Template

MIT_TEMPLATE = Template("""
** $model_id; version $model_version $attribution_website
$copyright_statement
 
MIT License

$copyright_statement

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")

def generate_thirdpart_statements_for_MIT(
    model_id: str,
    copyright_statement: str, 
    attribution_website: str, 
    model_version:str
) -> str:
    """
    Generate statements text for MIT-licensed third party model. The result should be put in the final artifact.
    
    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param copyright_statement: MIT models copyright statement
    :type copyright_statement: string
    :param attribution_website: The project website for MIT licensed models
    :type attribution_website: string
    :param model_version: The model version for MIT licensed models
    :type model_version: string
    :return: Statements text for MIT-licensed third party model.
    :rtype: str
    """
    
    
    result = MIT_TEMPLATE.substitute(model_id=model_id, copyright_statement=copyright_statement,
                                     attribution_website=attribution_website, model_version=model_version)
    return result.strip()