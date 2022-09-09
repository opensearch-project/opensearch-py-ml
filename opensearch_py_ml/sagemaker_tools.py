"""
Copyright OpenSearch Contributors
SPDX-License-Identifier: Apache-2.0
 """
import json

import numpy as np
from opensearch_py_ml import DataFrame
from typing import List, Optional, Dict, Tuple, Any
from math import ceil

from sagemaker import RealTimePredictor, Session

DEFAULT_SAGEMAKER_UPLOAD_CHUNK_SIZE = 1000


def make_sagemaker_prediction(endpoint_name: str,
                              data: DataFrame,
                              target_column: str,
                              sagemaker_session: Optional[Session] = None,
                              column_order: Optional[List[str]] = None,
                              chunksize: int = None,
                              sort_index: Optional[str] = '_doc'
                              ) -> Tuple[List[Any], Dict[Any, Any]]:
    """
    Make a prediction on an opensearch_py_ml dataframe using a deployed SageMaker model endpoint.

    Note that predictions will be returned based on the order in which data is ordered when
    ed.Dataframe.iterrows() is called on them.

    Parameters
    ----------
    endpoint_name: string representing name of SageMaker endpoint
    data: opensearch_py_ml DataFrame representing data to feed to SageMaker model. The dataframe must match the input datatypes
        of the model and also have the correct number of columns.
    target_column: column name of the dependent variable in the data.
    sagemaker_session: A SageMaker Session object, used for SageMaker interactions (default: None). If not specified,
        one is created using the default AWS configuration chain.
    column_order: list of string values representing the proper order that the columns of independent variables should
    be read into the SageMaker model. Must be a permutation of the column names of the opensearch_py_ml DataFrame.
    chunksize: how large each chunk being uploaded to sagemaker should be.
    sort_index: the index with which to sort the predictions by. Defaults to '_doc', an internal identifier for
        Lucene that optimizes performance.

    Returns
    ----------
    list representing the indices, dictionary representing the output of the model on input data
    """
    predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sagemaker_session, content_type='text/csv')
    data = data.drop(columns=target_column)

    if column_order is not None:
        data = data[column_order]
    if chunksize is None:
        chunksize = DEFAULT_SAGEMAKER_UPLOAD_CHUNK_SIZE

    indices = [index for index, _ in data.iterrows(sort_index=sort_index)]

    to_return = []

    for i in range(ceil(data.shape[0] / chunksize)):
        df_slice = indices[chunksize * i: min(len(indices), chunksize * (i+1))]
        to_process = data.filter(df_slice, axis=0)
        preds = predictor.predict(to_process.to_csv(header=False, index=False))
        to_return.append(preds)

    return indices, to_return
