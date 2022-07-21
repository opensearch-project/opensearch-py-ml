import json

import numpy as np
from eland import DataFrame
from typing import List, Optional
from math import ceil

from sagemaker import RealTimePredictor

DEFAULT_UPLOAD_CHUNK_SIZE = 1000


def make_sagemaker_prediction(endpoint_name: str,
                              data: DataFrame,
                              column_order: Optional[List[str]] = None,
                              chunksize: int = None
                              )-> np.array:
    """
    Make a prediction on an eland dataframe using a deployed SageMaker model endpoint.

    Note that predictions will be returned based on the order in which data is ordered when
    ed.Dataframe.iterrows() is called on them.

    Parameters
    ----------
    endpoint_name: string representing name of SageMaker endpoint
    data: eland DataFrame representing data to feed to SageMaker model. The dataframe must match the input datatypes
        of the model and also have the correct number of columns.
    column_order: list of string values representing the proper order that the columns should be read into the
        SageMaker model. Must be a permutation of the column names of the eland DataFrame.
    chunksize: how large each chunk being uploaded to sagemaker should be.

    Returns
    ----------
    np.array representing the output of the model on input data
    """
    predictor = RealTimePredictor(endpoint=endpoint_name, content_type='text/csv')
    test_data = data

    if column_order is not None:
        test_data = test_data[column_order]
    if chunksize is None:
        chunksize = DEFAULT_UPLOAD_CHUNK_SIZE

    indices = [index for index, _ in data.iterrows()]

    to_return = []

    for i in range(ceil(data.shape[0] / chunksize)):
        df_slice = indices[chunksize * i: min(len(indices), chunksize * (i+1))]

        to_process = test_data.filter(df_slice, axis=0)
        preds = predictor.predict(to_process.to_csv(header=False, index=False))
        preds = np.array(json.loads(preds.decode('utf-8'))['probabilities'])
        to_return.append(preds)

    return np.concatenate(to_return, axis=0)