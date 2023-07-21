# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" workflow
# (See model_uploader.yml) to update CHANGELOG.md after uploading the model
# to our model hub.

import argparse

from mdutils.fileutils import MarkDownFile

CHANGELOG_DIRNAME = "."
CHANGELOG_FILENAME = "CHANGELOG.md"
SECTION_NAME = "Added"


def update_changelog_file(
    changelog_line: str,
) -> None:
    """
    Update supported_models.json

    :param changelog_line: Line to be added to CHANGELOG.md
    :type changelog_line: string
    """
    changelog_data = MarkDownFile.read_file(f"{CHANGELOG_DIRNAME}/{CHANGELOG_FILENAME}")

    this_version_ptr = changelog_data.find("## [")
    assert this_version_ptr != -1, "Cannot find a version section in the CHANGELOG.md"
    next_version_ptr = changelog_data.find("## [", this_version_ptr + 1)
    this_version_section = changelog_data[this_version_ptr:next_version_ptr]

    this_subsection_ptr = this_version_section.find(f"### {SECTION_NAME}")
    if this_subsection_ptr != -1:
        next_subsection_ptr = this_version_section.find("### ", this_subsection_ptr + 1)
        this_subsection = this_version_section[
            this_subsection_ptr:next_subsection_ptr
        ].strip()
        this_subsection += "\n- " + changelog_line + "\n\n"
        new_version_section = (
            this_version_section[:this_subsection_ptr]
            + this_subsection
            + this_version_section[next_subsection_ptr:]
        )
    else:
        this_subsection = this_version_section.strip()
        this_subsection += "\n\n" + f"### {SECTION_NAME}\n- " + changelog_line + "\n\n"
        new_version_section = this_subsection

    new_changelog_data = (
        changelog_data[:this_version_ptr]
        + new_version_section
        + changelog_data[next_version_ptr:]
    )

    mdFile = MarkDownFile(CHANGELOG_FILENAME, dirname=CHANGELOG_DIRNAME)
    mdFile.rewrite_all_file(data=new_changelog_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "changelog_line",
        type=str,
        help="Line to be added to CHANGELOG.md",
    )
    args = parser.parse_args()
    update_changelog_file(args.changelog_line)
