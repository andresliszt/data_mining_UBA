# -*- coding: utf-8 -*-
"""Preprocessing."""


from typing import Union
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import haversine, Unit
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature

from outliers import FieldOutliers

# TODO: Agregar más data externa, como precio por metro cuadrado


class PreProcessing:
    """First instance of data preprocessing"""

    UNUSEFUL_COLUMNS = [
        "id",
        "l4",
        "l5",
        "l6",
        "price_period",
        "ad_type",
        "start_date",
        "end_date",
        "created_on",
    ]
    """Columns that do not contribute information to the model."""

    ALL_COLUMNS_TO_BE_ELIMINATED = [
        *UNUSEFUL_COLUMNS,
        "l1",
        "l2",
        "currency",
        "operation_type",
    ]
    """This columns will be eliminated after preprocessing. Should be eliminated in test data also"""

    ARG_TO_USD = 0.0085
    """Conversion from Argentine peso to USD."""
    # TODO: GET method into API

    @staticmethod
    def filter_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Filter the features that are considered in the model."""

        return data[
            (data.operation_type == "Venta")
            & (data.l1 == "Argentina")
            & (data.l2 == "Capital Federal")
            & ((data.property_type == "Departamento") | (data.l2 == "PH"))
        ]

    def transform_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform to USD and removes very low prices."""

        if not {"currency", "price"} <= set(data.columns):
            raise ValueError(
                "Este método necesita que data tenga 'price' y 'currency_type'"
            )

        data = data[~data.price.isnull() & ~data.currency.isnull()]
        data = data[data.price > FieldOutliers.RIDICULOUS_PRICE_FOR_PROPERTY]

        if not set(data.currency.unique()) <= {"ARS", "USD"}:
            raise ValueError(
                "Se debe tener como valores únicos en 'currency' a 'ARS' y 'USD'"
            )
        mask = data.currency == "ARS"
        data.loc[mask, "price"] = data.loc[mask, "price"].apply(
            lambda price: price * self.ARG_TO_USD
        )
        return data

    @staticmethod
    def check_is_in_polygon(
        geojson: pd.DataFrame,
        lat: Union[float, np.float32],
        lon: Union[float, np.float32],
    ) -> pd.Series:
        """Check if property with (lat, lon) is inside CABA

        Geojson file is needed, can be downloaded from
        ``https://data.buenosaires.gob.ar/dataset/barrios``
        The file is Barrios(GeoJson) and should be read
        using ``pd.read_json``.

        """

        # We ensure float type because np.float32 (comming from datafram) is not allowed for Point
        point = Feature(geometry=Point((float(lat), float(lon))))
        for _, row in geojson.iterrows():
            polygon = Polygon(
                [[[cord[1], cord[0]] for cord in row.geometry["coordinates"][0][0]]]
            )
            if boolean_point_in_polygon(point, polygon):
                return pd.Series(
                    [int(float(row.properties["COMUNA"])), row.properties["BARRIO"]]
                )
        return pd.Series([16, "Fuera de CABA"])

    @staticmethod
    def __make_centroides_per_l3(data: pd.DataFrame) -> Dict[str, float]:
        """Creates the centroids by locality ``l3``"""

        return {
            name: group[["lat", "lon"]].sum() / len(group)
            for name, group in data.groupby("l3")
        }

    def input_l3_latitude_longitude(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inputa latitud y longitud usando centroide de localidad l3

        To input the missing values in ``l3`` we have chosen
        fill in with the modal value, and to fill in the values
        missing ``lat`` and ``lon`` of a property in some neighborhood
        of 'l3' we have chosen to take the centroid of all
        properties located in that neighborhood.

        """

        # TODO: If l3 is missing but lat and lot not, I could use the polygon instead

        data.l3.fillna(data.l3.mode()[0], inplace=True)
        centroides = self.__make_centroides_per_l3(data)
        mask = data.lat.isnull() & data.lon.isnull()
        data.loc[mask, ["lat", "lon"]] = data.loc[mask, "l3"].apply(
            lambda loc: centroides[loc]
        )
        return data

    @staticmethod
    def __is_near(
        latitude: float, longitude: float, subte_cords: List[Tuple[float, float]]
    ) -> bool:
        for s_cord in subte_cords:
            if haversine((latitude, longitude), s_cord, unit=Unit.METERS) < 500:
                return True
        return False

    def near_to_subte(
        self, data: pd.DataFrame, subte_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Computa la distancia entre propiedades y estaciones del subte

        Usando la librería `harvsine` para calcular distancias entre
        pares de coordenadas, se calcula la distancia por propiedad
        a cada una de las estaciones del subte disponible en la data
        pública ``https://data.buenosaires.gob.ar/dataset/bocas-subte``.

        Se considera que una propiedad está cerca del subte, si es que
        está a menos de 500 metros de este.

        """

        print(
            "Calculando si las prop están cerca del subte... Esto puede tomar su tiempo"
        )

        subte_cords = [
            (rec["lat"], rec["lon"])
            for rec in subte_data[["lat", "lon"]].to_dict("records")
        ]
        data["near_subte"] = data.apply(
            lambda rec: self.__is_near(rec.lat, rec.lon, subte_cords), axis=1
        )
        return data

    def make_comune_and_neighborhood(
        self, data: pd.DataFrame, geojson: pd.DataFrame
    ) -> pd.DataFrame:
        """Create ``Comuna`` and ``Barrio`` columns using geojson file

        The lat and lon columns are used to see which commune
        and neighborhood of CABA the property belongs to.
        This is because ``l3`` can have unreliable values

        """
        print(
            "Construyendo Comunas y Barrios con archivo geojson... Esto puede tomar su tiempo"
        )
        data[["Comuna", "Barrio"]] = data.apply(
            lambda x: self.check_is_in_polygon(geojson, x.lat, x.lon), axis=1
        )
        # Comunas out de CABA las eliminamos (mejor opción? I Don't Know)
        data = data[~(data.Comuna == 16)]
        return data

    def _process(
        self,
        data: pd.DataFrame,
        subte_data: pd.DataFrame,
        geojson: pd.DataFrame,
        clean_data_path: Path,
        eliminate_outliers: bool,
    ) -> pd.DataFrame:
        data = data.copy()
        data = self.near_to_subte(
            self.make_comune_and_neighborhood(
                self.input_l3_latitude_longitude(
                    self.transform_prices(self.filter_columns(data))
                ),
                geojson=geojson,
            ),
            subte_data=subte_data,
        )

        data = (
            FieldOutliers().iqr_outliers_prices_per_comuna(data)
            if eliminate_outliers
            else data
        )
        data.to_csv(clean_data_path, index=False)
        return data

    def preprocess(
        self,
        data: pd.DataFrame,
        geojson: pd.DataFrame,
        subte_data: pd.DataFrame,
        clean_data_path: Path = Path("data_clean.csv"),
        eliminate_outliers: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Main preprocessing."""

        if force:
            return self._process(
                data,
                geojson=geojson,
                subte_data=subte_data,
                clean_data_path=clean_data_path,
                eliminate_outliers=eliminate_outliers,
            )

        try:
            data = pd.read_csv(clean_data_path)
            print(
                f"A data already processed was found in {clean_data_path}. Use ``force = True`` or change ``clean_data_path`` path."
            )
            return data
        except FileNotFoundError:
            print(
                f"Performing preprocessing transformations. Cache file not found in {clean_data_path}"
            )
            return self._process(
                data,
                geojson=geojson,
                subte_data=subte_data,
                clean_data_path=clean_data_path,
                eliminate_outliers=eliminate_outliers,
            )
