# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import re
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import opensearchpy
import pandas as pd  # type: ignore
from opensearchpy import OpenSearch

from ._version import __version__ as _opensearch_py_ml_version

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

# Default number of rows displayed (different to pandas where ALL could be displayed)
DEFAULT_NUM_ROWS_DISPLAYED = 60
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CSV_BATCH_OUTPUT_SIZE = 10000
DEFAULT_PROGRESS_REPORTING_NUM_ROWS = 10000
DEFAULT_SEARCH_SIZE = 5000
DEFAULT_PIT_KEEP_ALIVE = "3m"
DEFAULT_PAGINATION_SIZE = 5000  # for composite aggregations
PANDAS_VERSION: Tuple[int, ...] = tuple(
    int(part) for part in pd.__version__.split(".") if part.isdigit()
)[:2]

_OPENSEARCH_PY_ML_MAJOR_VERSION = int(_opensearch_py_ml_version.split(".")[0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    EMPTY_SERIES_DTYPE = pd.Series().dtype


def build_pd_series(
    data: Dict[str, Any], dtype: Optional["DTypeLike"] = None, **kwargs: Any
) -> pd.Series:
    """Builds a pd.Series while squelching the warning
    for unspecified dtype on empty series
    """
    dtype = dtype or (EMPTY_SERIES_DTYPE if not data else dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return pd.Series(data, **kwargs)


def docstring_parameter(*sub: Any) -> Callable[[Any], Any]:
    def dec(obj: Any) -> Any:
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


class SortOrder(Enum):
    ASC = 0
    DESC = 1

    @staticmethod
    def reverse(order: "SortOrder") -> "SortOrder":
        if order == SortOrder.ASC:
            return SortOrder.DESC

        return SortOrder.ASC

    @staticmethod
    def to_string(order: "SortOrder") -> str:
        if order == SortOrder.ASC:
            return "asc"

        return "desc"

    @staticmethod
    def from_string(order: str) -> "SortOrder":
        if order == "asc":
            return SortOrder.ASC

        return SortOrder.DESC


def opensearch_date_to_pandas_date(
    value: Union[int, str, float], date_format: Optional[str]
) -> pd.Timestamp:
    """
    Given a specific OpenSearch format for a date datatype, returns the
    'partial' `to_datetime` function to parse a given value in that format

    **Date Formats: https://opensearch.org/docs/2.2/opensearch/supported-field-types/date/

    Parameters
    ----------
    value: Union[int, str, float]
        The date value.
    date_format: str
        The OpenSearch date format (ex. 'epoch_millis', 'epoch_second', etc.)

    Returns
    -------
    datetime: pd.Timestamp
        Date formats can be customised, but if no format is specified then it uses the default:
        "strict_date_optional_time||epoch_millis"
        Therefore if no format is specified we assume either strict_date_optional_time
        or epoch_millis.
    """

    if date_format is None or isinstance(value, (int, float)):
        try:
            return pd.to_datetime(
                value, unit="s" if date_format == "epoch_second" else "ms"
            )
        except ValueError:
            return pd.to_datetime(value)
    elif date_format == "epoch_millis":
        return pd.to_datetime(value, unit="ms")
    elif date_format == "epoch_second":
        return pd.to_datetime(value, unit="s")
    elif date_format == "strict_date_optional_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "basic_date":
        return pd.to_datetime(value, format="%Y%m%d")
    elif date_format == "basic_date_time":
        return pd.to_datetime(value, format="%Y%m%dT%H%M%S.%f", exact=False)
    elif date_format == "basic_date_time_no_millis":
        return pd.to_datetime(value, format="%Y%m%dT%H%M%S%z")
    elif date_format == "basic_ordinal_date":
        return pd.to_datetime(value, format="%Y%j")
    elif date_format == "basic_ordinal_date_time":
        return pd.to_datetime(value, format="%Y%jT%H%M%S.%f%z", exact=False)
    elif date_format == "basic_ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y%jT%H%M%S%z")
    elif date_format == "basic_time":
        return pd.to_datetime(value, format="%H%M%S.%f%z", exact=False)
    elif date_format == "basic_time_no_millis":
        return pd.to_datetime(value, format="%H%M%S%z")
    elif date_format == "basic_t_time":
        return pd.to_datetime(value, format="T%H%M%S.%f%z", exact=False)
    elif date_format == "basic_t_time_no_millis":
        return pd.to_datetime(value, format="T%H%M%S%z")
    elif date_format == "basic_week_date":
        return pd.to_datetime(value, format="%GW%V%u")
    elif date_format == "basic_week_date_time":
        return pd.to_datetime(value, format="%GW%V%uT%H%M%S.%f%z", exact=False)
    elif date_format == "basic_week_date_time_no_millis":
        return pd.to_datetime(value, format="%GW%V%uT%H%M%S%z")
    elif date_format == "strict_date":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "date":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "strict_date_hour":
        return pd.to_datetime(value, format="%Y-%m-%dT%H")
    elif date_format == "date_hour":
        return pd.to_datetime(value, format="%Y-%m-%dT%H")
    elif date_format == "strict_date_hour_minute":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M")
    elif date_format == "date_hour_minute":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M")
    elif date_format == "strict_date_hour_minute_second":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S")
    elif date_format == "date_hour_minute_second":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S")
    elif date_format == "strict_date_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "date_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "strict_date_hour_minute_second_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "date_hour_minute_second_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "strict_date_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "date_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S%z")
    elif date_format == "date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S%z")
    elif date_format == "strict_hour":
        return pd.to_datetime(value, format="%H")
    elif date_format == "hour":
        return pd.to_datetime(value, format="%H")
    elif date_format == "strict_hour_minute":
        return pd.to_datetime(value, format="%H:%M")
    elif date_format == "hour_minute":
        return pd.to_datetime(value, format="%H:%M")
    elif date_format == "strict_hour_minute_second":
        return pd.to_datetime(value, format="%H:%M:%S")
    elif date_format == "hour_minute_second":
        return pd.to_datetime(value, format="%H:%M:%S")
    elif date_format == "strict_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "hour_minute_second_fraction":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "strict_hour_minute_second_millis":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "hour_minute_second_millis":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "strict_ordinal_date":
        return pd.to_datetime(value, format="%Y-%j")
    elif date_format == "ordinal_date":
        return pd.to_datetime(value, format="%Y-%j")
    elif date_format == "strict_ordinal_date_time":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S.%f%z", exact=False)
    elif date_format == "ordinal_date_time":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S%z")
    elif date_format == "ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S%z")
    elif date_format == "strict_time":
        return pd.to_datetime(value, format="%H:%M:%S.%f%z", exact=False)
    elif date_format == "time":
        return pd.to_datetime(value, format="%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_time_no_millis":
        return pd.to_datetime(value, format="%H:%M:%S%z")
    elif date_format == "time_no_millis":
        return pd.to_datetime(value, format="%H:%M:%S%z")
    elif date_format == "strict_t_time":
        return pd.to_datetime(value, format="T%H:%M:%S.%f%z", exact=False)
    elif date_format == "t_time":
        return pd.to_datetime(value, format="T%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_t_time_no_millis":
        return pd.to_datetime(value, format="T%H:%M:%S%z")
    elif date_format == "t_time_no_millis":
        return pd.to_datetime(value, format="T%H:%M:%S%z")
    elif date_format == "strict_week_date":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "week_date":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "strict_week_date_time":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S.%f%z", exact=False)
    elif date_format == "week_date_time":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_week_date_time_no_millis":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S%z")
    elif date_format == "week_date_time_no_millis":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S%z")
    elif date_format == "strict_weekyear" or date_format == "weekyear":
        # TODO investigate if there is a way of converting this
        raise NotImplementedError(
            "strict_weekyear is not implemented due to support in pandas"
        )
        return pd.to_datetime(value, format="%G")
        # Not supported in pandas
        # ValueError: ISO year directive '%G' must be used with the ISO week directive '%V'
        # and a weekday directive '%A', '%a', '%w', or '%u'.
    elif date_format == "strict_weekyear_week" or date_format == "weekyear_week":
        # TODO investigate if there is a way of converting this
        raise NotImplementedError(
            "strict_weekyear_week is not implemented due to support in pandas"
        )
        return pd.to_datetime(value, format="%G-W%V")
        # Not supported in pandas
        # ValueError: ISO year directive '%G' must be used with the ISO week directive '%V'
        # and a weekday directive '%A', '%a', '%w', or '%u'.
    elif date_format == "strict_weekyear_week_day":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "weekyear_week_day":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "strict_year":
        return pd.to_datetime(value, format="%Y")
    elif date_format == "year":
        return pd.to_datetime(value, format="%Y")
    elif date_format == "strict_year_month":
        return pd.to_datetime(value, format="%Y-%m")
    elif date_format == "year_month":
        return pd.to_datetime(value, format="%Y-%m")
    elif date_format == "strict_year_month_day":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "year_month_day":
        return pd.to_datetime(value, format="%Y-%m-%d")
    else:
        warnings.warn(
            f"The '{date_format}' format is not explicitly supported."
            f"Using pandas.to_datetime(value) to parse value",
            Warning,
        )
        # TODO investigate how we could generate this just once for a bulk read.
        return pd.to_datetime(value)


def os_version(os_client: OpenSearch) -> Tuple[int, int, int]:
    """Tags the current OS client with a cached '_os_ml_py_version'
    property if one doesn't exist yet for the current OpenSearch version.
    """
    opensearch_py_ml_os_version: Tuple[int, int, int]
    if not hasattr(os_client, "_os_ml_py_version"):
        version_info = os_client.info()["version"]["number"]
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_info)
        if match is None:
            raise ValueError(
                f"Unable to determine version. " f"Received: {version_info}"
            )
        opensearch_py_ml_os_version = cast(
            Tuple[int, int, int], tuple(int(x) for x in match.groups())
        )
        os_client._os_ml_py_version = opensearch_py_ml_os_version  # type: ignore

        # Raise a warning if the major version of the library doesn't match the
        # the OpenSearch server major version.
        if opensearch_py_ml_os_version[0] != _OPENSEARCH_PY_ML_MAJOR_VERSION:
            warnings.warn(
                f"OpenSearch major version ({_opensearch_py_ml_version}) doesn't match the major "
                f"version of the OpenSearch server ({version_info}) which can lead "
                f"to compatibility issues. Your major version should be the same "
                "as your cluster major version.",
                stacklevel=2,
            )

    else:
        opensearch_py_ml_os_version = os_client._os_ml_py_version  # type: ignore
    return opensearch_py_ml_os_version


OPENSEARCH_HOST = "https://instance:9200"
OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD = "admin", "admin"

# Define client to use in tests
OPENSEARCH_TEST_CLIENT = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
    verify_certs=False,
)
# in github integration test, host url is: https://instance:9200
# in development, usually host url is: https://localhost:9200
# it's hard to remember changing the host url. So applied a try catch so that we don't have to keep change this config
try:
    OS_VERSION = os_version(OPENSEARCH_TEST_CLIENT)
except opensearchpy.exceptions.ConnectionError:
    OPENSEARCH_HOST = "https://localhost:9200"
    # Define client to use in tests
    OPENSEARCH_TEST_CLIENT = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
        verify_certs=False,
    )
    OS_VERSION = os_version(OPENSEARCH_TEST_CLIENT)
