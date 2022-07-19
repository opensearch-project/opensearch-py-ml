import json

import numpy as np
from eland import DataFrame
from typing import List, Optional

from sagemaker import RealTimePredictor


def make_sagemaker_prediction(endpoint_name: str,
                              data: DataFrame,
                              column_order: Optional[List[str]] = None
                              ) -> np.array:
    """
    Make a prediction on an eland dataframe using a deployed SageMaker model endpoint.

    Parameters
    ----------
    endpoint_name: string representing name of SageMaker endpoint
    data: eland DataFrame representing data to feed to SageMaker model. The dataframe must match the input datatypes
        of the model and also have the correct number of columns.
    column_order: list of string values representing the proper order that the columns should be read into the
        SageMaker model. Must be a permutation of the column names of the eland DataFrame.

    Returns
    ----------
    np.array representing the output of the model on input data
    """
    predictor = RealTimePredictor(endpoint=endpoint_name, content_type='text/csv')

    test_data = data
    if column_order is not None:
        test_data = test_data[column_order]

    preds = predictor.predict(test_data.to_csv(header=False, index=False))
    preds = np.array(json.loads(preds.decode('utf-8'))['probabilities'])
    return preds