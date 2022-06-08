# -*- coding: utf-8 -*-
"""Inputers."""


from __future__ import annotations
import re
import unicodedata
from typing import Union
from typing import List

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.impute._base import _BaseImputer

# pylint: disable=unused-argument

# TODO: Inputaciones de otro estilo como MICE, KNN, etc


class RoomsAndSurfaceInputer(_BaseImputer):
    """Impute missing values for surfacesm enviroments, bedrooms and bathrooms

    Based on heuristics. Note that the ``fit`` method does nothing,
    Strictly speaking, it should save all computations as attributes.
    what I do (how to calculate the mode) in the ``fit`` method, for later
    with the help of the ``memory`` argument to the Pipeline they don't have to
    recalculate. I think it would be too much for this job, because
    imputation methods are fast. All logic falls back on
    ``self.transform``

    """

    @staticmethod
    def __fillna_with_mode(
        data: pd.DataFrame, room_name: str, na_mask: pd.DataFrame, neq_na_mask=None
    ) -> pd.DataFrame:

        if neq_na_mask is None:
            neq_na_mask = ~na_mask

        data.loc[na_mask, room_name] = data.loc[na_mask, room_name].fillna(
            data[neq_na_mask][room_name].mode()[0]
        )
        return data

    def fillna_rooms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple inputer para rooms

        If a property has a non-null `bedrooms` field,
        but the rooms field (`rooms`) is null, so it is entered
        for the number of rooms plus one. For the case in which it is not
        have the number of rooms, simply use the value
        modal for environments.

        Observation:
            If this method is used after `self.fillna_bedrooms`, then
            the modal value will never be used.
        """
        # Mascara para bed rooms no nulos y rooms nulos
        mask = data.rooms.isna()
        mask_bedrooms = ~data.bedrooms.isna()
        # Si tenemos registros nulos para rooms, pero hay registros para bedrooms, entonces
        # Hacemos fillna con bedrooms + 1
        data.loc[mask & mask_bedrooms, "rooms"] = (
            data.loc[mask & mask_bedrooms, "bedrooms"] + 1
        )
        # En otro caso, reemplazamos con el valor modal
        # Re calculamos la mascara,pues ya fue suplida en algunos casos
        # TODO: Valor modal por barrio?
        return self.__fillna_with_mode(data, "rooms", data.rooms.isna())

    def fillna_bedrooms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple inputer para bedrooms

        If a property has a non-null `rooms` field,
        but the rooms field (`bedrooms`) is null, so it is entered
        by the number of environments minus one, or simply 1 if
        it is a studio apartment. For the case in which it is not
        have the number of environments, simply use the value
        modal for the rooms.

        Observation:
            If this method is used after `self.fillna_rooms`, then
            the modal value will never be used.
        """

        # Mascara para bed rooms
        mask = data.bedrooms.isna()
        mask_rooms = ~data.rooms.isna()
        # Si tenemos registros nulos para bedrooms, pero hay registros para rooms, entonces
        # Hacemos fillna con rooms - 1, Si es mono ambiente rooms = bedrooms
        data.loc[mask & mask_rooms, "bedrooms"] = data.loc[
            mask & mask_rooms, "rooms"
        ].apply(lambda room: room if room == 1 else room - 1)
        return self.__fillna_with_mode(data, "bedrooms", data.bedrooms.isna())

    @staticmethod
    def __rooms_mask(data: pd.DataFrame, n_rooms: int) -> List[pd.DataFrame]:
        if n_rooms < 1:
            raise ValueError("``n_rooms`` debe ser mayor a 1")
        masks = [data.rooms == room for room in range(1, n_rooms)]
        masks.append(data.rooms >= n_rooms)
        return masks

    def fillna_bathrooms(self, data: pd.DataFrame) -> pd.DataFrame:
        # Mascara para rooms que no son nulas
        masks_rooms = self.__rooms_mask(data, 3)

        for mask_room in masks_rooms:
            data = self.__fillna_with_mode(
                data,
                "bathrooms",
                data.bathrooms.isna() & mask_room,
                ~data.bathrooms.isna() & mask_room,
            )
        return data

    def fillna_surfaces(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill the NaNs for 'surface_total' and 'surface_covered'

        Based on heuristics. First look up the NaNs for 'surface_total'
        and completes them by the average value per number of rooms. By
        example, if a property with 2 rooms has 'surface_total' NaN,
        then all properties of 2 rooms for which
        if there is 'surface_total' and the average is calculated that will be the filler.

        If for 'surface_covered' NaNs, complete with 'surface_total'

        """
        # Llenamos los NaN de surface_total por el promedio de acuerdo a la cantidad de rooms
        # para rooms de 1, de 2, de 3 y las mayores a 4
        masks_rooms = self.__rooms_mask(data, 4)

        # TODO: Juntar este fillna con __fillna_with_mode en un solo método

        for mask_room in masks_rooms:
            mask = data.surface_total.isna()
            data.loc[mask & mask_room, "surface_total"] = data.loc[
                mask & mask_room, "surface_total"
            ].fillna(data[~mask & mask_room].surface_total.median())

        # Llenamos los NaN de surface_covered con el valor de surface_total
        data.surface_covered = data.surface_covered.fillna(data.surface_total)

        if "has_exterior" in data.columns:
            # Si la data ya viene con si tiene o balcon, retornamos data
            return data
        data["has_exterior"] = data.apply(
            lambda x: x.surface_total - x.surface_covered > 0, axis=1
        )
        print(
            "`has_exterior` fue construída analizando la diferencia entre sup total y sup cubierta"
        )

        return data

    @staticmethod
    def ensure_surface_total_is_greater_than_covered(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Si 'surface_covered' es mayor que 'surface_total', se reeplaza por la total"""
        mask = data.surface_covered > data.surface_total
        data.loc[mask, "surface_covered"] = data.loc[mask, "surface_total"]
        return data

    def fit(self, X, y=None) -> RoomsAndSurfaceInputer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return self.ensure_surface_total_is_greater_than_covered(
            self.fillna_surfaces(
                self.fillna_bathrooms(self.fillna_bedrooms(self.fillna_rooms(X)))
            )
        )


class FindMissingFields(_BaseImputer):
    """Try to find missing field from text

    The properties have a description and a title,
    when any of the fields (bathrooms, surface_total, bedrooms, etc.)
    come null, this class will try to be the first imputation barrier
    by using regular expressions to find the
    missing data in the text

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_keywords_regex = self.__field_keywords_regex()
        self.rooms_methods = {
            "bathrooms": self.find_bathrooms,
            "bedrooms": self.find_bedrooms,
            "rooms": self.find_rooms,
        }

    # TODO: Regex para encontrar la superficie

    STEMMER = SnowballStemmer("spanish")

    HTML_TAGS_REGEX = re.compile(r"<.*?>|\\n")
    """Regex para eliminar los tags html desde el texto"""

    SPECIAL_CHARACTERS_REGEX = re.compile(r"(?!<\d)[\.\,](?!\d)|[^A-Za-z0-9*.,]+")
    """Regex para caracteres especiales. Siempre debe usarse al final"""

    BAD_SUBSTRINGS = ["tilde", "acute"]
    """Palabras extrañas de parseo html a eliminar"""

    FIELD_KEYWORDS = {
        "bathrooms": ["ban", "toilet", "toilett"],
        "bedrooms": ["dormitori", "piez"],
        "rooms": ["ambient", "monoambient"],
        "surface": ["mt2", "m2", "mts2"],
    }

    NUMBER_AS_STRING = {
        "uno": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
    }

    def stem(self, word) -> str:
        if word in self.NUMBER_AS_STRING:
            # No queremos hacer stemmer a los números como string!
            return word
        return self.STEMMER.stem(word)

    def __field_keywords_regex(self):
        keyw_regex_dict = {}
        for field, keyw_list in self.FIELD_KEYWORDS.items():
            keyw_regex_dict[field] = [
                re.compile(rf"([\S]+)(?= \b{keyw}\b)") for keyw in keyw_list
            ]
        return keyw_regex_dict

    @staticmethod
    def strip_accents(text):
        return str(
            unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
        )

    def regex_sub(self, text: str) -> str:
        text = self.strip_accents(text)
        text = re.sub(self.HTML_TAGS_REGEX, "", text)
        return " ".join(
            [
                self.stem(re.sub(self.SPECIAL_CHARACTERS_REGEX, "", word).lower())
                for word in text.split(" ")
            ]
        )

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        for bad_substring in self.BAD_SUBSTRINGS:
            text = text.replace(bad_substring, "")
        return self.regex_sub(text)

    @staticmethod
    def _find_left_side_of(
        text: str, keywords: List[re.Pattern], search_type: str = "first"
    ) -> List[str]:
        occurrences = []
        if search_type == "first":
            for regex in keywords:
                match = regex.findall(text)
                if match:
                    occurrences.append(match[0])
        elif search_type == "all":
            for regex in keywords:
                for match in regex.findall(text):
                    occurrences.append(match)
        else:
            raise NotImplementedError(
                "'search_type' definido solo para 'first' y 'all'"
            )

        return occurrences

    def left_side_to_number(self, word: str) -> int:
        try:
            return int(word)
        except ValueError:
            try:
                return self.NUMBER_AS_STRING[word]
            except KeyError:
                return 1

    def find_parts_of_the_house(self, text, keywords, search_type) -> Union[int, float]:
        # Esto es para asegurar de que si el keyword viene al inicio del string
        text = "- " + text
        left_side_of_keywords = self._find_left_side_of(text, keywords, search_type)
        if not left_side_of_keywords:
            # En este caso no se encontro match, por ende dejamos el valor en np.nan
            return np.nan
        return sum(
            [self.left_side_to_number(left_word) for left_word in left_side_of_keywords]
        )

    def find_bathrooms(self, text: str) -> Union[int, float]:
        n_bathrooms = self.find_parts_of_the_house(
            text,
            keywords=self.field_keywords_regex["bathrooms"],
            search_type="all",
        )
        if n_bathrooms > 3:
            # Aseguremonos de evitar posibles errores
            return 1
        return n_bathrooms

    def find_bedrooms(self, text: str) -> Union[int, float]:
        n_bedrooms = self.find_parts_of_the_house(
            text,
            keywords=self.field_keywords_regex["bedrooms"],
            search_type="first",
        )
        if n_bedrooms > 4:
            # Aseguremonos de evitar posibles errores
            return 1
        return n_bedrooms

    def find_rooms(self, text: str) -> Union[int, float]:
        n_rooms = self.find_parts_of_the_house(
            text,
            keywords=self.field_keywords_regex["rooms"],
            search_type="first",
        )
        if n_rooms > 5:
            # Aseguremonos de evitar posibles errores
            return 1
        return n_rooms

    def find_surface(self, text: str) -> Union[int, float]:
        return self.find_parts_of_the_house(
            text,
            keywords=self.field_keywords_regex["surface"],
            search_type="first",
        )

    def fit(self, X, y=None) -> FindMissingFields:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
