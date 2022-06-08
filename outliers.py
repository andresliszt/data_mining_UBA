# -*- coding: utf-8 -*-
"""Deteccíón e imputación de outliers"""

from __future__ import annotations

import pandas as pd

# pylint: disable=unused-argument
# TODO: Otras técnicas de imputación de outliers, KNN, 2D para lat, lon, etc.


class FieldOutliers:
    """Elimination of outliers for different fields of the dataset.

    The idea is to implement here different logics
    field-dependent outlier removal to which it will apply.
    For now it is implemented the IQR.

    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    # TODO: Tal vez implementar eliminación de outliers 2D para latitud y longitud
    # TODO: En los inputers tengo replacement por moda, mean, etc podría disponibilizar métodos desde acá

    RIDICULOUS_PRICE_FOR_PROPERTY = 40000

    def iqr_outliers(
        self, data: pd.DataFrame, field: str, strategy="mode", margin: float = 1.5
    ) -> pd.DataFrame:
        """Método de remoción de outliers usando IQR"""

        # Será la primera barrera de eliminación, solo tenemos campos positivos
        if not data[data[field].isnull()].empty:
            raise RuntimeError(
                f"Para aplicar IQR asegurate que el campo '{field}' no contenga NaN"
            )

        Q1 = data[field].quantile(0.25)
        Q3 = data[field].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~(
            (data[field] < (Q1 - margin * IQR)) | (data[field] > (Q3 + margin * IQR))
        )

        if self.replace:
            if strategy == "mode":
                replace = data[field].mode()[0]
            elif strategy == "mean":
                replace = data[field].mean()
            else:
                replace = data[field].median()
            data.loc[mask, field] = data.loc[mask, field].fillna(replace)
            return data

        return data[mask]

    def iqr_outliers_prices_per_comuna(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtra outliers por cada comuna 'l3'

        The idea behind removing localized outliers by 'l3' is the socioeconomic
        level, the prices of communes more exclusive areas,
        such as Puerto Madero or Recoleta, would be affected when
        trimming with global IQR. A possible better one is maybe
        do not cut by 'l3', but by larger areas, like south zone, central zone
        and north zone (It is a TODO!)
        """

        data = data[data.price > self.RIDICULOUS_PRICE_FOR_PROPERTY]
        groups = [
            self.iqr_outliers(group, "price")
            for _, group in data.copy().groupby("Comuna")
        ]
        return pd.concat(groups)

    def fit(self, X: pd.DataFrame, y=None) -> FieldOutliers:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self.iqr_outliers(X, "rooms", margin=3.2)
        X = self.iqr_outliers(X, "bedrooms", margin=3.2)
        X = self.iqr_outliers(X, "bathrooms", margin=3.2)
        X = self.iqr_outliers(X, "surface_total", strategy="mean")
        return X
