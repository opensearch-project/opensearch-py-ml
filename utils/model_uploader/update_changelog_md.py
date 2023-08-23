# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading"
# & "Model Listing Uploading" workflow (See model_uploader.yml
# & model_listing_uploader.yml) to update CHANGELOG.md after
# uploading the model to our model hub.

import argparse

from mdutils.fileutils import MarkDownFile

CHANGELOG_DIRNAME = "."
CHANGELOG_FILENAME = "CHANGELOG.md"
SUBSECTION_NAME = "Changed"
PREV_SUBSECTION_NAME = "Added"


def update_changelog_file(
    changelog_line: str,
) -> None:
    """
    Update supported_models.json

    :param changelog_line: Line to be added to CHANGELOG.md
    :type changelog_line: string
    :return: No return value expected
    :rtype: None
    """
    changelog_data = MarkDownFile.read_file(f"{CHANGELOG_DIRNAME}/{CHANGELOG_FILENAME}")

    # Find the most recent version section and pull it out
    this_version_ptr = changelog_data.find("\n## ") + 1
    assert this_version_ptr != 0, "Cannot find a version section in the CHANGELOG.md"
    next_version_ptr = changelog_data.find("\n## ", this_version_ptr + 1) + 1
    if next_version_ptr == 0:
        next_version_ptr = -1
    this_version_section = changelog_data[this_version_ptr:next_version_ptr]

    # Find the sub-section SUBSECTION_NAME
    this_subsection_ptr = this_version_section.find(f"\n### {SUBSECTION_NAME}") + 1
    if this_subsection_ptr != 0:
        # Case 1: Section SUBSECTION_NAME exists
        # Append a change_log line to the end of that subsection if it exists
        next_subsection_ptr = (
            this_version_section.find("\n### ", this_subsection_ptr + 1) + 1
        )
        if next_subsection_ptr == 0:
            next_subsection_ptr = -1
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
        # Case 2: Sub-section SUBSECTION_NAME does not exist
        # Create sub-section SUBSECTION_NAME and add a change_log line
        this_subsection = f"### {SUBSECTION_NAME}\n- {changelog_line}\n\n"
        prev_subsection_ptr = (
            this_version_section.find(f"\n### {PREV_SUBSECTION_NAME}") + 1
        )
        if prev_subsection_ptr != 0:
            # Case 2.1: Sub-section PREV_SUBSECTION_NAME exist
            # Add a sub-section SUBSECTION_NAME after PREV_SUBSECTION_NAME if PREV_SUBSECTION_NAME exists
            next_subsection_ptr = (
                this_version_section.find("\n### ", prev_subsection_ptr + 1) + 1
            )
            prev_subsection = this_version_section[
                prev_subsection_ptr:next_subsection_ptr
            ].strip()
            new_version_section = (
                this_version_section[:prev_subsection_ptr]
                + prev_subsection
                + "\n\n"
                + this_subsection
                + this_version_section[next_subsection_ptr:]
            )
        else:
            # Case 2.2: Sub-section PREV_SUBSECTION_NAME does not exist
            next_subsection_ptr = this_version_section.find("\n### ") + 1
            if next_subsection_ptr != 0:
                # Case 2.2.1: There exists other sub-section in this version section
                # Add a sub-section SECTION_NAME before other sub-sections
                new_version_section = (
                    this_version_section[:next_subsection_ptr]
                    + this_subsection
                    + this_version_section[next_subsection_ptr:]
                )
            else:
                # Case 2.2.2: There isn't any other sub-section in this version section
                # Add a sub-section SECTION_NAME after version headline
                new_version_section = (
                    this_version_section.strip() + "\n\n" + this_subsection
                )

    # Insert new_version_section back to the document
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
