# -*- coding: utf-8 -*-
"""Pipelines for model execution."""

from __future__ import annotations
from typing import List


import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from outliers import FieldOutliers
from inputers import RoomsAndSurfaceInputer
from pre_processing import PreProcessing

# pylint: disable=unused-argument


class ColumnDropper:
    """Remove some column(s) inside pipeline

    This helper class can be useful when a column
    is required within a pipeline stage, but
    for the rest no. A specific example where it is
    used is to remove ``surface_covered``, as this
    column is used inside the pipeline to build
    a new feature ``has_exterior`` to identify
    if the apartment has an exterior or not, once built
    ``surface_covered`` is deleted.

    """

    def __init__(self, columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None) -> ColumnDropper:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop)


def pipeline_for_prepare_data_for_input_model() -> Pipeline:
    """Prepare the data to train a model

    The idea that as development proceeds, preprocessing
    must be sophisticated. For now we do inputations of the fields
    of type rooms with ``inputers.RoosInputer``

    """

    # TODO: Crear argumentos para cambiar quizas los Scalers, etc.
    # TODO: Meter superficies a Outliers

    fields = Pipeline(
        [
            ("rooms_outliers", FieldOutliers(replace=True)),
            ("drop_surface_covered", ColumnDropper(columns_to_drop="surface_covered")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "fields",
                fields,
                ["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"],
            ),
            (
                "scale",
                MinMaxScaler(),
                ["surface_total", "lat", "lon"],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Comuna", "property_type"],
            ),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("rooms_inputers", RoomsAndSurfaceInputer()),
            ("transformer", column_transformer),
        ]
    )


def prepare_model_for_grid_search(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", pipeline_for_prepare_data_for_input_model()),
            ("regression", model()),
        ]
    )


def prepare_test_data(
    test_data: pd.DataFrame, subte_data: pd.DataFrame
) -> pd.DataFrame:
    """It simply removes the columns that were removed in the preprocess.

    ``test_data`` should be the test csv to upload to kaggle.

    """
    test_data = PreProcessing().near_to_subte(test_data, subte_data)

    return test_data.drop(
        columns=[
            "details",
            "additionals",
            "services",
            "price",
            *PreProcessing.ALL_COLUMNS_TO_BE_ELIMINATED,
        ]
    )
