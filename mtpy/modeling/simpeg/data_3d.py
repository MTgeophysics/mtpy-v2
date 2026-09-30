# =============================================================================
# Imports
# =============================================================================
import warnings

import numpy as np
from simpeg.electromagnetics import natural_source as nsem

warnings.filterwarnings("ignore")


# =============================================================================


class Simpeg3DData:
    """ """

    def __init__(self, dataframe, **kwargs):
        # mtpy (EDI) components as keys, x = north, y = east, z = down, and
        # the SimPEG component each becomes, x = east, y = north, z = up.
        # Swapping x and y swaps the indices of every element. Flipping z also
        # changes the sign of the tipper, H_z / H_h, which ``to_rec_array``
        # applies (see ``_negated_components``).
        self.component_map = {
            "z_xx": {"simpeg": "zyy", "z+": "z_xx"},
            "z_xy": {"simpeg": "zyx", "z+": "z_xy"},
            "z_yx": {"simpeg": "zxy", "z+": "z_yx"},
            "z_yy": {"simpeg": "zxx", "z+": "z_yy"},
            "t_zx": {"simpeg": "tzy", "z+": "t_zx"},
            "t_zy": {"simpeg": "tzx", "z+": "t_zy"},
        }

        self._component_list = list(self.component_map.keys())

        self._rec_columns = {
            "frequency": "freq",
            "east": "x",
            "north": "y",
            "elevation": "z",
            "model_east": "x",
            "model_north": "y",
            "model_elevation": "z",
        }
        self._rec_columns.update(
            dict(
                [
                    (key, self.component_map[key]["simpeg"])
                    for key in self._component_list
                ]
            )
        )

        self._rec_dtypes = [
            ("freq", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("zxx", complex),
            ("zxy", complex),
            ("zyx", complex),
            ("zyy", complex),
            ("tzx", complex),
            ("tzy", complex),
        ]

        self.include_elevation = False
        self.topography = None  # should be a geotiff or asc file
        self.geographic_coordinates = True
        self.invert_z_xx = True
        self.invert_z_xy = True
        self.invert_z_yx = True
        self.invert_z_yy = True
        self.invert_t_zx = True
        self.invert_t_zy = True
        self.invert_types = ["real", "imaginary"]

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dataframe = dataframe

    @property
    def station_locations(self):
        """
        return just station locations in appropriate coordinates, default is
        geographic.
        """
        station_df = self.dataframe.groupby("station").nth(0)
        if self.geographic_coordinates:
            if self.include_elevation:
                return np.c_[station_df.east, station_df.north, station_df.elevation]
            else:
                return np.c_[
                    station_df.east,
                    station_df.north,
                    np.zeros(station_df.elevation.size),
                ]
        else:
            if self.include_elevation:
                return np.c_[
                    station_df.model_east,
                    station_df.model_north,
                    station_df.model_elevation,
                ]
            else:
                return np.c_[
                    station_df.model_east,
                    station_df.model_north,
                    np.zeros(station_df.model_elevation.size),
                ]

    @property
    def frequencies(self):
        """unique frequencies from the dataframe"""

        return 1.0 / self.dataframe.period.unique()

    @property
    def n_frequencies(self):
        return self.frequencies.size

    @property
    def n_stations(self):
        return self.dataframe.station.unique().size

    @property
    def station_names(self):
        """list of station names"""
        return list(self.dataframe.station.unique())

    @property
    def components_to_invert(self):
        """get a list of components to invert base on user input"""
        components = []
        for comp in self.component_map.keys():
            if getattr(self, f"invert_{comp}"):
                components.append(comp)
        return components

    @property
    def _rec_components_to_invert(self) -> list[str]:
        """get list of rec array keys to invert"""
        return ["freq", "x", "y", "z"] + [
            self._rec_columns[comp] for comp in self.components_to_invert
        ]

    @property
    def _rec_dtype_to_invert(self) -> list[tuple]:
        """get dtype of rec array keys to invert"""
        rec_dtype = []
        for comp in self._rec_components_to_invert:
            for rec_type in self._rec_dtypes:
                if rec_type[0] == comp:
                    rec_dtype.append(rec_type)
                    break
        return rec_dtype

    @property
    def n_orientation(self):
        """number of components to invert"""
        return len(self.components_to_invert)

    def _get_z_mode_sources(self, orientation):
        """Get the source for each mode"""
        source_real = nsem.receivers.Impedance(
            self.station_locations, orientation=orientation, component="real"
        )
        source_imag = nsem.receivers.Impedance(
            self.station_locations, orientation=orientation, component="imag"
        )

        return [source_real, source_imag]

    def _get_t_mode_sources(self, orientation):
        """Get the source for each mode"""
        source_real = nsem.receivers.Tipper(
            self.station_locations, orientation=orientation, component="real"
        )
        source_imag = nsem.receivers.Tipper(
            self.station_locations, orientation=orientation, component="imag"
        )

        return [source_real, source_imag]

    def _simpeg_orientation(self, component):
        """SimPEG receiver orientation of an mtpy component, e.g. ``"yx"`` for
        ``"z_xy"``."""
        return self.component_map[component]["simpeg"][1:]

    @property
    def source_z_xx(self):
        """receivers for mtpy z_xx [simpeg yy]"""
        return self._get_z_mode_sources(self._simpeg_orientation("z_xx"))

    @property
    def source_z_xy(self):
        """receivers for mtpy z_xy [simpeg yx]"""
        return self._get_z_mode_sources(self._simpeg_orientation("z_xy"))

    @property
    def source_z_yx(self):
        """receivers for mtpy z_yx [simpeg xy]"""
        return self._get_z_mode_sources(self._simpeg_orientation("z_yx"))

    @property
    def source_z_yy(self):
        """receivers for mtpy z_yy [simpeg xx]"""
        return self._get_z_mode_sources(self._simpeg_orientation("z_yy"))

    @property
    def source_t_zx(self):
        """receivers for mtpy t_zx [simpeg zy, negated]"""
        return self._get_t_mode_sources(self._simpeg_orientation("t_zx"))

    @property
    def source_t_zy(self):
        """receivers for mtpy t_zy [simpeg zx, negated]"""
        return self._get_t_mode_sources(self._simpeg_orientation("t_zy"))

    @property
    def _sources_list(self):
        rx_list = []

        for comp in self.components_to_invert:
            rx_list += getattr(self, f"source_{comp}")

        return rx_list

    @property
    def sources(self):
        return [
            nsem.sources.PlanewaveXYPrimary(self._sources_list, frequency=f)
            for f in self.frequencies
        ]

    @property
    def survey(self):
        """returns a survey object with the requested components"""
        return nsem.Survey(self.sources)

    def get_survey(self, index):
        """get a survey object for a particular frequency index

        Useful for using `MetaClass`

        :param index: _description_
        :type index: _type_
        :return: _description_
        :rtype: _type_
        """
        return nsem.Survey(self.sources[index])

    @property
    def _negated_components(self):
        """mtpy components whose SimPEG value is the negative: the tipper, a
        ratio with H_z, which changes sign with the z axis."""
        return ["t_zx", "t_zy"]

    def to_rec_array(self):
        cols = ["period"]
        if self.geographic_coordinates:
            cols += ["east", "north", "elevation"]
        else:
            cols += ["model_east", "model_north", "model_elevation"]

        df = self.dataframe[cols + self.components_to_invert].copy()
        for comp in self._negated_components:
            if comp in df.columns:
                df[comp] = -df[comp]

        df.loc[:, "frequency"] = 1.0 / df.period.to_numpy()
        df = df.drop(columns="period")
        new_column_names = [self._rec_columns[col] for col in df.columns]
        df.columns = new_column_names
        df = df[new_column_names]

        return df.to_records(index=False, column_dtypes=dict(self._rec_dtype_to_invert))

    def standard_deviations(self):
        """get model errors for the data"""
        df = self.dataframe[[f"{comp}_model_error" for comp in self.component_map]]

    def get_simpeg_data_object(self) -> nsem.Data:
        """create a data object"""
        nsem_data = nsem.Data(self.survey)

        return nsem_data.fromRecArray(self.to_rec_array())

    def from_simpeg_data_object(self, data_object):
        """ingest a Simpeg.data.Data object into a dataframe

        :param data_object: _description_
        :type data_object: _type_
        """
        raise NotImplementedError()
