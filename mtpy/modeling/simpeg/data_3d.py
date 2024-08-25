# =============================================================================
# Imports
# =============================================================================
import warnings
import numpy as np

from simpeg.electromagnetics import natural_source as nsem
from simpeg import data

warnings.filterwarnings("ignore")


# =============================================================================


class Simpeg3DData:
    """ """

    def __init__(self, dataframe, **kwargs):
        

        # nez+ as keys then enz- as values
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
        if self.geographic_coordinates:
            station_df = self.dataframe.groupby("station").nth(0)

            if self.include_elevation:
                return np.c_[
                    station_df.east, station_df.north, station_df.elevation
                ]
            else:
                return np.c_[
                    station_df.east,
                    station_df.north,
                    np.zeros(station_df.elevation.size),
                ]

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
        """ list of station names """
        return list(self.dataframe.station.unique())
    
    @property
    def components_to_invert(self):
        """ get a list of components to invert base on user input"""
        components = []
        for comp in self.component_map.keys():
            if getattr(self, f"invert_{comp}"):
                components.append(comp)
        return components
    
    @property
    def n_orientation(self):
        """ number of components to invert"""
        return len(self.components_to_invert)

    def _get_z_mode_sources(self, orientation):
        """Get the source for each mode"""
        source_real = nsem.receivers.PointNaturalSource(
            self.station_locations, orientation=orientation, component="real"
        )
        source_imag = nsem.receivers.PointNaturalSource(
            self.station_locations, orientation=orientation, component="imag"
        )

        return [source_real, source_imag]

    def _get_t_mode_sources(self, orientation):
        """Get the source for each mode"""
        source_real = nsem.receivers.Point3DTipper(
            self.station_locations, orientation=orientation, component="real"
        )
        source_imag = nsem.receivers.Point3DTipper(
            self.station_locations, orientation=orientation, component="imag"
        )

        return [source_real, source_imag]

    @property
    def source_z_xx(self):
        """xx source [simpeg xx -> nez+ yy]"""
        return self._get_z_mode_sources("xx")

    @property
    def source_z_xy(self):
        """xy source [simpeg xy -> nez+ yx]"""
        return self._get_z_mode_sources("xy")

    @property
    def source_z_yx(self):
        """yx source [simpeg yx -> nez+ xy]"""
        return self._get_z_mode_sources("yx")

    @property
    def source_z_yy(self):
        """yy source [simpeg yy -> nez+ xx]"""
        return self._get_z_mode_sources("yy")

    @property
    def source_t_zx(self):
        """zx source [simpeg zx -> nez+ zy]"""
        return self._get_t_mode_sources("zx")

    @property
    def source_t_zy(self):
        """zy source [simpeg zy -> nez+ zx]"""
        return self._get_t_mode_sources("zy")

    @property
    def _sources_list(self):
        rx_list = []

        for comp in self._component_list:
            if getattr(self, f"invert_{comp}"):
                rx_list += getattr(self, f"source_{comp}")

        return rx_list

    @property
    def sources(self):
        rx_list = self.rx_list
        return [
            nsem.sources.PlanewaveXYPrimary(rx_list, frequency=f)
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

    def to_rec_array(self):

        df = self.dataframe[
            ["period", "east", "north", "elevation"] + self._component_list
        ]

        df.loc[:, "frequency"] = 1.0 / df.period.to_numpy()
        df = df.drop(columns="period")
        new_column_names = [self._rec_columns[col] for col in df.columns]
        df.columns = new_column_names
        df = df[[col[0] for col in self._rec_dtypes]]

        return df.to_records(index=False, column_dtypes=dict(self._rec_dtypes))

    def get_observations_and_erros(self):
        """ build object from a dataframe """

        # inverting for real and imaginary
        n_component = len(self.invert_types)

        f_dict = dict([(round(ff, 5), ii) for ii, ff in enumerate(self.frequencies)])
        observations = np.zeros((self.n_frequencies, self.n_orientation, n_component, self.n_stations))
        errors = np.zeros_like(observations)
        for s_index, station in enumerate(self.station_names):
            station_df = self.dataframe.loc[self.dataframe.station == station]
            station_df.set_index("period", inplace=True)
            for row in station_df.itertuples():
                f_index = f_dict[round(1./row.Index, 5)]
                for c_index, comp in enumerate(self.components_to_invert):
                    value = getattr(row, comp)
                    err = getattr(row, f"{comp}_model_error") # user_set error
                    # err = getattr(row, f"{comp}_error") # measurement error from data statistics
                    observations[f_index, c_index, 0, s_index] = value.real
                    observations[f_index, c_index, 1, s_index] = value.imag
                    errors[f_index, c_index, 0, s_index] = err
                    errors[f_index, c_index, 1, s_index] = err

        observations[np.where(np.nan_to_num(observations) == 0)] = 100
        errors[np.where(np.nan_to_num(errors) == 0)] = np.inf
         
        return observations, errors
    
    @property
    def data_object(self):
        """ create a data object"""
        observations, errors = self.get_observations_and_erros()
        survey = self.get_survey()
        return data.Data(survey, 
                         dobs=observations.flatten(),
                           standard_deviation=errors.flatten())