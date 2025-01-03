"""
==================
ModEM
==================

# Generate files for ModEM

# revised by JP 2017
# revised by AK 2017 to bring across functionality from ak branch
# revised by JP 2021 updating functionality and updating docs

"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np
import xarray as xr
from scipy import stats as stats
from scipy import interpolate
from loguru import logger

import mtpy.modeling.mesh_tools as mtmesh
import mtpy.modeling.gocad as mtgocad
import mtpy.utils.calculator as mtcc
import mtpy.utils.filehandling as mtfh

from mtpy.utils.gis_tools import project_point
from mtpy.modeling.plots.plot_mesh import PlotMesh
from mtpy.core.mt_location import MTLocation
from mtpy.gis.raster_tools import array2raster

from pyevtk.hl import gridToVTK

# =============================================================================


class StructuredGrid3D:
    """Make and read a FE mesh grid

    The mesh assumes the coordinate system where:
        x == North
        y == East
        z == + down

    All dimensions are in meters.

    The mesh is created by first making a regular grid around the station area,
    then padding cells are added that exponentially increase to the given
    extensions.  Depth cell increase on a log10 scale to the desired depth,
    then padding cells are added that increase exponentially..

    Arguments:
            **station_object** : mtpy.modeling.modem.Stations object
                                .. seealso:: mtpy.modeling.modem.Stations

    Examples:

        :Example 1 --> create mesh first then data file: ::

            >>> import mtpy.modeling.modem as modem
            >>> import os
            >>> # 1) make a list of all .edi files that will be inverted for
            >>> edi_path = r"/home/EDI_Files"
            >>> edi_list = [os.path.join(edi_path, edi)
                            for edi in os.listdir(edi_path)
            >>> ...         if edi.find('.edi') > 0]
            >>> # 2) Make a Stations object
            >>> stations_obj = modem.Stations()
            >>> stations_obj.get_station_locations_from_edi(edi_list)
            >>> # 3) make a grid from the stations themselves with 200m cell spacing
            >>> mmesh = modem.Model(station_obj)
            >>> # change cell sizes
            >>> mmesh.cell_size_east = 200,
            >>> mmesh.cell_size_north = 200
            >>> mmesh.ns_ext = 300000 # north-south extension
            >>> mmesh.ew_ext = 200000 # east-west extension of model
            >>> mmesh.make_mesh()
            >>> # check to see if the mesh is what you think it should be
            >>> msmesh.plot_mesh()
            >>> # all is good write the mesh file
            >>> msmesh.write_model_file(save_path=r"/home/modem/Inv1")
            >>> # create data file
            >>> md = modem.Data(edi_list, station_locations=mmesh.station_locations)
            >>> md.write_data_file(save_path=r"/home/modem/Inv1")

        :Example 2 --> Rotate Mesh: ::

            >>> mmesh.mesh_rotation_angle = 60
            >>> mmesh.make_mesh()

        .. note:: ModEM assumes all coordinates are relative to North and East, and
                 does not accommodate mesh rotations, therefore, here the rotation
                 is of the stations, which essentially does the same thing.  You
                 will need to rotate you data to align with the 'new' coordinate
                 system.

        ==================== ======================================================
        Attributes           Description
        ==================== ======================================================
        _logger              python logging object that put messages in logging
                             format defined in logging configure file, see MtPyLog
                             more information
        cell_number_ew       optional for user to specify the total number of sells
                             on the east-west direction. *default* is None
        cell_number_ns       optional for user to specify the total number of sells
                             on the north-south direction. *default* is None
        cell_size_east       mesh block width in east direction
                             *default* is 500
        cell_size_north      mesh block width in north direction
                             *default* is 500
        grid_center          center of the mesh grid
        grid_east            overall distance of grid nodes in east direction
        grid_north           overall distance of grid nodes in north direction
        grid_z               overall distance of grid nodes in z direction
        model_fn             full path to initial file name
        model_fn_basename    default name for the model file name
        n_air_layers         number of air layers in the model. *default* is 0
        n_layers             total number of vertical layers in model
        nodes_east           relative distance between nodes in east direction
        nodes_north          relative distance between nodes in north direction
        nodes_z              relative distance between nodes in east direction
        pad_east             number of cells for padding on E and W sides
                             *default* is 7
        pad_north            number of cells for padding on S and N sides
                             *default* is 7
        pad_num              number of cells with cell_size with outside of
                             station area.  *default* is 3
        pad_method           method to use to create padding:
                             extent1, extent2 - calculate based on ew_ext and
                             ns_ext
                             stretch - calculate based on pad_stretch factors
        pad_stretch_h        multiplicative number for padding in horizontal
                             direction.
        pad_stretch_v        padding cells N & S will be pad_root_north**(x)
        pad_z                number of cells for padding at bottom
                             *default* is 4
        ew_ext               E-W extension of model in meters
        ns_ext               N-S extension of model in meters
        res_scale            scaling method of res, supports
                               'loge' - for log e format
                               'log' or 'log10' - for log with base 10
                               'linear' - linear scale
                             *default* is 'loge'
        res_list             list of resistivity values for starting model
        res_model            starting resistivity model
        res_initial_value    resistivity initial value for the resistivity model
                             *default* is 100
        mesh_rotation_angle  Angle to rotate the grid to. Angle is measured
                             positve clockwise assuming North is 0 and east is 90.
                             *default* is None
        save_path            path to save file to
        sea_level            sea level in grid_z coordinates. *default* is 0
        station_locations    location of stations
        title                title in initial file
        z1_layer             first layer thickness
        z_bottom             absolute bottom of the model *default* is 300,000
        z_target_depth       Depth of deepest target, *default* is 50,000
        ==================== ======================================================
    """

    def __init__(self, station_locations=None, center_point=None, **kwargs):
        self._logger = logger

        self.station_locations = None
        self.center_point = MTLocation()

        if station_locations is not None:
            self.station_locations = station_locations

        if center_point is not None:
            self.center_point = center_point
            self.model_epsg = self.center_point.utm_epsg

        # size of cells within station area in meters
        self.cell_size_east = 500
        self.cell_size_north = 500

        # FZ: added this for user input number of cells in the horizontal mesh
        self.cell_number_ew = None
        self.cell_number_ns = None

        # padding cells on either side
        self.pad_east = 7
        self.pad_north = 7
        self.pad_z = 4

        self.pad_num = 3

        self.ew_ext = 100000
        self.ns_ext = 100000

        # root of padding cells
        self.pad_stretch_h = 1.2
        self.pad_stretch_v = 1.2

        self.z1_layer = 10
        self.z_layer_rounding = 0
        self.z_target_depth = 50000
        self.z_bottom = 300000

        # number of vertical layers
        self.n_layers = 30

        # number of air layers
        self.n_air_layers = 0
        # sea level in grid_z coordinates. Auto adjusts when topography read in?
        self.sea_level = 0.0

        # strike angle to rotate grid to
        self.mesh_rotation_angle = 0

        # --> attributes to be calculated
        # grid nodes
        self._nodes_east = None
        self._nodes_north = None
        self._nodes_z = None

        # grid locations
        self.grid_east = None
        self.grid_north = None
        self.grid_z = kwargs.pop("grid_z", None)
        if self.grid_z is not None:
            self.n_layers = len(self.grid_z)
            self.z_mesh_method = "custom"
        else:
            self.z_mesh_method = "default"
        if "z_mesh_method" in list(kwargs.keys()):
            self.z_mesh_method = kwargs["z_mesh_method"]

        # method to use to create padding
        self.pad_method = "extent1"

        self.grid_center = None
        self.surface_dict = {}

        # resistivity model
        self.res_initial_value = 100.0
        self.res_model = None

        # initial file stuff
        self.save_path = Path().cwd()
        self.model_fn_basename = "ModEM_Model_File.rho"

        self._modem_title = "Model File written by MTpy.modeling.modem"
        self.res_scale = "loge"

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._logger.warning(
                    f"Argument {key}={value} is not supportted thus not been set."
                )

    def __str__(self):
        """Str function."""
        lines = ["Structured3DMesh Model Object:", "-" * 20]
        # --> print out useful information
        try:
            lines.append(
                f"\tNumber of stations = {len(self.station_locations.station)}"
            )
        except AttributeError:
            lines.append("\tNumber of stations = unknown")

        lines.append("\tMesh Parameter: ")
        lines.append(f"\t\tcell_size_east:    {self.cell_size_east}")
        lines.append(f"\t\tcell_size_north:   {self.cell_size_north}")
        lines.append(f"\t\tpad_east:          {self.pad_east}")
        lines.append(f"\t\tpad_north:         {self.pad_north}")
        lines.append(f"\t\tpad_num:           {self.pad_num}")
        lines.append(f"\t\tz1_layer:          {self.z1_layer}")
        lines.append(f"\t\tz_target_depth:    {self.z_target_depth}")
        lines.append(f"\t\tn_layers:          {self.n_layers}")
        lines.append(f"\t\tn_air_layers:      {self.n_air_layers}")
        lines.append(f"\t\tres_initial_value: {self.res_initial_value}")
        lines.append("\tDimensions: ")
        lines.append(f"\t\te-w: {self.grid_east.size}")
        lines.append(f"\t\tn-s: {self.grid_north.size}")
        lines.append(f"\t\tz:   {self.grid_z.size} (without 7 air layers)")
        lines.append("\tExtensions: ")
        lines.append(f"\t\te-w:  {self.nodes_east.__abs__().sum():.1f} (m)")
        lines.append(f"\t\tn-s:  {self.nodes_north.__abs__().sum():.1f} (m)")
        lines.append(f"\t\t0-z:  {self.nodes_z.__abs__().sum():.1f} (m)")
        if self.mesh_rotation_angle != 0:
            lines.append(
                f"\tStations rotated by: {self.mesh_rotation_angle:.1f} deg clockwise positive from N"
            )

            lines.append(
                " ** Note rotations are assumed to have stations rotated."
            )
            lines.append("    All coordinates are aligned to geographic N, E")
            lines.append(
                "    therefore rotating the stations will have a similar effect"
            )
            lines.append("    as rotating the mesh.")
        lines.append("-" * 20)
        return "\n".join(lines)

    def __repr__(self):
        """Repr function."""
        return self.__str__()

    @property
    def save_path(self):
        """Save path."""
        return self._save_path

    @save_path.setter
    def save_path(self, save_path):
        """Save path."""
        if save_path is None:
            self._save_path = Path().cwd()
        else:
            self._save_path = Path(save_path)

        if not self._save_path.exists():
            self._save_path.mkdir()

    @property
    def model_fn(self):
        """Model fn."""
        return self.save_path.joinpath(self.model_fn_basename)

    @model_fn.setter
    def model_fn(self, filename):
        """Model fn."""
        if filename is not None:
            filename = Path(filename)
            self.save_path = filename.parent
            self.model_fn_basename = filename.name

    @property
    def model_epsg(self):
        """Model epsg."""
        return self.center_point.utm_epsg

    @model_epsg.setter
    def model_epsg(self, value):
        """Model epsg."""
        self.center_point.utm_epsg = value

    # --> make nodes and grid symbiotic so if you set one the other one
    #     gets set as well
    # Nodes East
    @property
    def nodes_east(self):
        """Nodes east."""
        if self.grid_east is not None:
            self._nodes_east = np.array(
                [
                    abs(self.grid_east[ii + 1] - self.grid_east[ii])
                    for ii in range(self.grid_east.size - 1)
                ]
            )
        return self._nodes_east

    @nodes_east.setter
    def nodes_east(self, nodes):
        """Nodes east."""
        nodes = np.array(nodes)
        self._nodes_east = nodes
        self.grid_east = np.array(
            [
                nodes[0:ii].sum() for ii in range(nodes.size + 1)
            ]  # -nodes.sum() / 2 +
        )  # + [shift])#[nodes.sum() / 2]

    # Nodes North
    @property
    def nodes_north(self):
        """Nodes north."""
        if self.grid_north is not None:
            self._nodes_north = np.array(
                [
                    abs(self.grid_north[ii + 1] - self.grid_north[ii])
                    for ii in range(self.grid_north.size - 1)
                ]
            )
        return self._nodes_north

    @nodes_north.setter
    def nodes_north(self, nodes):
        """Nodes north."""
        nodes = np.array(nodes)
        self._nodes_north = nodes
        self.grid_north = np.array(
            [
                nodes[0:ii].sum() for ii in range(nodes.size + 1)
            ]  # -nodes.sum() / 2 +
        )  # + [shift])#[nodes.sum() / 2]

    @property
    def nodes_z(self):
        """Nodes z."""
        if self.grid_z is not None:
            self._nodes_z = np.array(
                [
                    abs(self.grid_z[ii + 1] - self.grid_z[ii])
                    for ii in range(self.grid_z.size - 1)
                ]
            )

            return self._nodes_z

    @nodes_z.setter
    def nodes_z(self, nodes):
        """Nodes z."""
        nodes = np.array(nodes)
        self._nodes_z = nodes
        self.grid_z = np.array(
            [nodes[0:ii].sum() for ii in range(nodes.size)] + [nodes.sum()]
        )

    # need some arrays for plotting that are the same length as the
    # resistivity model
    @property
    def plot_east(self):
        """Plot east."""
        plot_east = np.array(
            [self.nodes_east[0:ii].sum() for ii in range(self.nodes_east.size)]
        )
        return plot_east - plot_east[-1] / 2.0

    @property
    def plot_north(self):
        """Plot north."""
        plot_north = np.array(
            [
                self.nodes_north[0:ii].sum()
                for ii in range(self.nodes_north.size)
            ]
        )
        return plot_north - plot_north[-1] / 2.0

    @property
    def plot_z(self):
        """Plot z."""
        return np.array(
            [self.nodes_z[0:ii].sum() for ii in range(self.nodes_z.size)]
        )

    def make_mesh(self, verbose=True):
        """Create finite element mesh according to user-input parameters.

        The mesh is built by:

            1. Making a regular grid within the station area.
              - Uses `cell_size_east` and `cell_size_north` for dimensions
            2. Adding `pad_num` of cell_width cells outside of station area
            3. Adding padding cells to given extension and number of padding
               cells.
               - `extent1` - stretch to a given distance with `pad_east` or
                `pad_north` number of cells.
               - `extent2` - stretch to a given distance with `pad_east` or
                `pad_north` number of cells.
               - `stretch` stretches from station area using
                `pad_north` and `pad_east` times `pad_stretch_h`
            4. Making vertical cells starting with z1_layer increasing
               logarithmically (base 10) to z_target_depth and num_layers.
               - `default` creates a vertical mesh that increases
                logarithmically down.  See `make_z_mesh`.
               - `custom` input your own vertical mesh.
            5. Add vertical padding cells to desired extension.
            6. Check to make sure none of the stations lie on a node.
               If they do then move the node by .02*cell_width
        """

        # --> find the edges of the grid
        # calculate the extra width of padding cells
        # multiply by 1.5 because this is only for 1 side
        pad_width_east = self.pad_num * 1.5 * self.cell_size_east
        pad_width_north = self.pad_num * 1.5 * self.cell_size_north

        # get the extremities
        west = self.station_locations.model_east.min() - pad_width_east
        east = self.station_locations.model_east.max() + pad_width_east
        south = self.station_locations.model_north.min() - pad_width_north
        north = self.station_locations.model_north.max() + pad_width_north

        # round the numbers so they are easier to read
        west = np.round(west, -2)
        east = np.round(east, -2)
        south = np.round(south, -2)
        north = np.round(north, -2)

        # -------make a grid around the stations from the parameters above------

        # adjust the edges so we have a whole number of cells
        add_ew = ((east - west) % self.cell_size_east) / 2.0
        add_ns = ((north - south) % self.cell_size_north) / 2.0

        # --> make the inner grid first
        inner_east = np.arange(
            west + add_ew - self.cell_size_east,
            east - add_ew + 2 * self.cell_size_east,
            self.cell_size_east,
        )
        inner_north = np.arange(
            south + add_ns + self.cell_size_north,
            north - add_ns + 2 * self.cell_size_north,
            self.cell_size_north,
        )

        # compute padding cells
        # first validate ew_ext and ns_ext to ensure it is large enough
        if "extent" in self.pad_method:
            self._validate_extent(
                inner_east.min(),
                inner_east.max(),
                inner_north.min(),
                inner_north.max(),
            )

        if self.pad_method == "extent1":
            padding_east = mtmesh.get_padding_cells(
                self.cell_size_east,
                self.ew_ext / 2 - east,
                self.pad_east,
                self.pad_stretch_h,
            )
            padding_north = mtmesh.get_padding_cells(
                self.cell_size_north,
                self.ns_ext / 2 - north,
                self.pad_north,
                self.pad_stretch_h,
            )
        elif self.pad_method == "extent2":
            padding_east = mtmesh.get_padding_cells2(
                self.cell_size_east,
                inner_east[-1],
                self.ew_ext / 2.0,
                self.pad_east,
            )
            padding_north = mtmesh.get_padding_cells2(
                self.cell_size_north,
                inner_north[-1],
                self.ns_ext / 2.0,
                self.pad_north,
            )
        elif self.pad_method == "stretch":
            padding_east = mtmesh.get_padding_from_stretch(
                self.cell_size_east, self.pad_stretch_h, self.pad_east
            )
            padding_north = mtmesh.get_padding_from_stretch(
                self.cell_size_north, self.pad_stretch_h, self.pad_north
            )
        else:
            raise NameError(
                f'Padding method "{self.pad_method}" is not supported'
            )

        # make the horizontal grid
        self.grid_east = np.append(
            np.append(-1 * padding_east[::-1] + inner_east.min(), inner_east),
            padding_east + inner_east.max(),
        )
        self.grid_north = np.append(
            np.append(
                -1 * padding_north[::-1] + inner_north.min(), inner_north
            ),
            padding_north + inner_north.max(),
        )

        # --> need to make sure none of the stations lie on the nodes
        for s_east in sorted(self.station_locations.model_east):
            try:
                node_index = np.where(
                    abs(s_east - self.grid_east) < 0.02 * self.cell_size_east
                )[0][0]
                if s_east - self.grid_east[node_index] > 0:
                    self.grid_east[node_index] -= 0.02 * self.cell_size_east
                elif s_east - self.grid_east[node_index] < 0:
                    self.grid_east[node_index] += 0.02 * self.cell_size_east
            except IndexError:
                continue

        # --> need to make sure none of the stations lie on the nodes
        for s_north in sorted(self.station_locations.model_north):
            try:
                node_index = np.where(
                    abs(s_north - self.grid_north)
                    < 0.02 * self.cell_size_north
                )[0][0]
                if s_north - self.grid_north[node_index] > 0:
                    self.grid_north[node_index] -= 0.02 * self.cell_size_north
                elif s_north - self.grid_north[node_index] < 0:
                    self.grid_north[node_index] += 0.02 * self.cell_size_north
            except IndexError:
                continue

        if self.z_mesh_method == "custom":
            if self.grid_z is None:
                self.z_mesh_method = "default"
                self._logger.warning(
                    "No grid_z provided, creating new z mesh using default method"
                )

        if self.z_mesh_method == "custom":
            self.nodes_z, z_grid = (
                self.grid_z[1:] - self.grid_z[:-1],
                self.grid_z,
            )
        elif self.z_mesh_method == "default":
            self.nodes_z, z_grid = self.make_z_mesh()
        else:
            raise NameError(
                f'Z mesh method "{self.z_mesh_method}" is not supported'
            )

        # compute grid center
        center_east = np.round(
            self.grid_east.min() - self.grid_east.mean(), -1
        )
        center_north = np.round(
            self.grid_north.min() - self.grid_north.mean(), -1
        )
        center_z = 0

        # this is the value to the lower left corner from the center.
        self.grid_center = np.array([center_north, center_east, center_z])

        # make the resistivity array
        self.res_model = np.zeros(
            (self.nodes_north.size, self.nodes_east.size, self.nodes_z.size)
        )
        self.res_model[:, :, :] = self.res_initial_value

        # --> print out useful information
        if verbose:
            print(self.__str__())

    def make_z_mesh(self, n_layers=None):
        """New version of make_z_mesh. make_z_mesh and M."""
        n_layers = self.n_layers if n_layers is None else n_layers

        # --> make depth grid
        # if n_airlayers < 0; set to 0
        log_z = mtcc.make_log_increasing_array(
            self.z1_layer, self.z_target_depth, n_layers - self.pad_z
        )

        if self.z_layer_rounding is not None:
            z_nodes = np.around(log_z, decimals=self.z_layer_rounding)
        else:
            # round any values less than 100 to the same s.f. as z1_layer
            z_nodes = np.around(
                log_z[log_z < 100],
                decimals=-int(np.floor(np.log10(self.z1_layer))),
            )
            # round any values greater than or equal to 100 to the nearest 100
            z_nodes = np.append(
                z_nodes, np.around(log_z[log_z >= 100], decimals=-2)
            )

        # index of top of padding
        # itp = len(z_nodes) - 1

        # padding cells in the vertical direction
        z_0 = float(z_nodes[-1])
        for ii in range(1, self.pad_z + 1):
            pad_d = np.round(z_0 * self.pad_stretch_v**ii, -2)
            z_nodes = np.append(z_nodes, pad_d)
        # add air layers and define ground surface level.
        # initial layer thickness is same as z1_layer
        # z_nodes = np.hstack([[z1_layer] * n_air, z_nodes])

        # make an array of absolute values
        z_grid = np.array(
            [z_nodes[:ii].sum() for ii in range(z_nodes.shape[0] + 1)]
        )

        return z_nodes, z_grid

    def add_layers_to_mesh(
        self, n_add_layers=None, layer_thickness=None, where="top"
    ):
        """Function to add constant thickness layers to the top or bottom of mesh.

        Note: It is assumed these layers are added before the topography. If
        you want to add topography layers, use function add_topography_to_model
        :param n_add_layers: Integer, number of layers to add, defaults to None.
        :param layer_thickness: Real value or list/array. Thickness of layers,
             Can provide a single value
            or a list/array containing multiple layer
            thicknesses, defaults to None.
        :param where: Where to add, top or bottom, defaults to "top".
        """
        # create array containing layers to add
        if layer_thickness is None:
            layer_thickness = self.z1_layer
        if np.iterable(layer_thickness):
            add_layers = np.insert(np.cumsum(layer_thickness), 0, 0)[:-1]
            layer_thickness = layer_thickness[-1]

            if n_add_layers != len(add_layers):
                self._logger.warning(
                    "Updating number of layers to reflect the length of the layer thickness array"
                )
            n_add_layers = len(add_layers)
        else:
            add_layers = np.arange(
                0, n_add_layers * layer_thickness, layer_thickness
            )

        # create a new z grid
        self.grid_z = np.hstack(
            [add_layers, self.grid_z + add_layers[-1] + layer_thickness]
        )

        # update the number of layers
        self.n_layers = len(self.grid_z) - 1

        # add the extra layer to the res model
        self.res_model = np.vstack(
            [self.res_model[:, :, :n_add_layers].T, self.res_model.T]
        ).T

    def assign_resistivity_from_surface_data(
        self, top_surface, bottom_surface, resistivity_value
    ):
        """Assign resistivity value to all points above or below a surface
        requires the surface_dict attribute to exist and contain data for
        surface key (can get this information from ascii file using
        project_surface)

        **inputs**
        surface_name = name of surface (must correspond to key in surface_dict)
        resistivity_value = value to assign
        where = 'above' or 'below' - assign resistivity above or below the
                surface
        """

        # FZ: should ref-define the self.res_model if its shape has changed after topo air layer are added

        gcz = np.mean([self.grid_z[:-1], self.grid_z[1:]], axis=0)

        self._logger.debug(
            f"gcz is the cells centre coordinates: {len(gcz)}, {gcz}"
        )

        # assign resistivity value
        for j in range(len(self.res_model)):
            for i in range(len(self.res_model[j])):
                ii = np.where(
                    (gcz > top_surface[j, i]) & (gcz <= bottom_surface[j, i])
                )[0]
                self.res_model[j, i, ii] = resistivity_value

    def to_modem(self, model_fn=None, **kwargs):
        """Will write an initial file for ModEM.

        Note that x is assumed to be S --> N, y is assumed to be W --> E and
        z is positive downwards.  This means that index [0, 0, 0] is the
        southwest corner of the first layer.  Therefore if you build a model
        by hand the layer block will look as it should in map view.

        Also, the xgrid, ygrid and zgrid are assumed to be the relative
        distance between neighboring nodes.  This is needed because wsinv3d
        builds the  model from the bottom SW corner assuming the cell width
        from the init file.

        Key Word Arguments::


                **model_fn_basename** : string
                                        basename to save file to
                                        *default* is ModEM_Model.ws
                                        file is saved at save_path/model_fn_basename

                **title** : string
                            Title that goes into the first line
                            *default* is Model File written by MTpy.modeling.modem

                **res_starting_value** : float
                                         starting model resistivity value,
                                         assumes a half space in Ohm-m
                                         *default* is 100 Ohm-m

                **res_scale** : [ 'loge' | 'log' | 'log10' | 'linear' ]
                                scale of resistivity.  In the ModEM code it
                                converts everything to Loge,
                                *default* is 'loge'
        """
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

        # get resistivity model
        if self.res_model is None:
            self.res_model = np.zeros(
                (
                    self.nodes_north.size,
                    self.nodes_east.size,
                    self.nodes_z.size,
                )
            )
            self.res_model[:, :, :] = self.res_initial_value

        # --> write file
        lines = []

        lines.append(f"# {self._modem_title.upper()}")
        lines.append(
            f"{self.nodes_north.size:>5}{self.nodes_east.size:>5}"
            f"{self.nodes_z.size:>5}{0:>5} {self.res_scale.upper()}"
        )

        # write S --> N node block
        lines.append(
            "".join([f"{abs(nnode):>12.3f}" for nnode in self.nodes_north])
        )

        # write W --> E node block
        lines.append(
            "".join([f"{abs(enode):>12.3f}" for enode in self.nodes_east])
        )

        # write top --> bottom node block
        lines.append(
            "".join([f"{abs(znode):>12.3f}" for znode in self.nodes_z])
        )

        # write the resistivity in log e format
        if self.res_scale.lower() == "loge":
            write_res_model = np.log(self.res_model[::-1, :, :])
        elif (
            self.res_scale.lower() == "log"
            or self.res_scale.lower() == "log10"
        ):
            write_res_model = np.log10(self.res_model[::-1, :, :])
        elif self.res_scale.lower() == "linear":
            write_res_model = self.res_model[::-1, :, :]
        else:
            raise ValueError(
                f'resistivity scale "{self.res_scale}" is not supported.'
            )

        # write out the layers from resmodel
        for zz in range(self.nodes_z.size):
            lines.append("")
            for ee in range(self.nodes_east.size):
                line = []
                for nn in range(self.nodes_north.size):
                    line.append(f"{write_res_model[nn, ee, zz]:>13.5E}")
                lines.append("".join(line))

        if self.grid_center is None:
            # compute grid center
            center_east = -self.nodes_east.__abs__().sum() / 2
            center_north = -self.nodes_north.__abs__().sum() / 2
            center_z = 0
            self.grid_center = np.array([center_north, center_east, center_z])

        lines.append("")
        lines.append(
            f"{self.grid_center[0]:>16.3f}{self.grid_center[1]:>16.3f}{self.grid_center[2]:>16.3f}"
        )

        if self.mesh_rotation_angle is None:
            lines.append(f"{0:>9.3f}")
        else:
            lines.append(f"{self.mesh_rotation_angle:>9.3f}")

        if model_fn is not None:
            self.model_fn = model_fn

        with open(self.model_fn, "w") as ifid:
            ifid.write("\n".join(lines))

        self._logger.info(f"Wrote file to: {self.model_fn}")

    def from_modem(self, model_fn=None):
        """Read an initial file and return the pertinent information including
        grid positions in coordinates relative to the center point (0,0) and
        starting model.

        Note that the way the model file is output, it seems is that the
        blocks are setup as

        ModEM:                           WS::
            0-----> N_north                 0-------->N_east
            |                               |
            |                               |
            V                               V
            N_east                          N_north

        Arguments::

                **model_fn** : full path to initializing file.

        Outputs::

                **nodes_north** : np.array(nx)
                            array of nodes in S --> N direction

                **nodes_east** : np.array(ny)
                            array of nodes in the W --> E direction

                **nodes_z** : np.array(nz)
                            array of nodes in vertical direction positive downwards

                **res_model** : dictionary
                            dictionary of the starting model with keys as layers

                **res_list** : list
                            list of resistivity values in the model

                **title** : string
                             title string
        """

        if model_fn is not None:
            self.model_fn = model_fn

        if self.model_fn is None:
            raise ValueError("model_fn is None, input a model file name")

        if not self.model_fn.exists():
            raise ValueError(f"Cannot find {self.model_fn}, check path")

        with open(self.model_fn, "r") as ifid:
            ilines = ifid.readlines()

        self._modem_title = ilines[0].replace("#", "").strip()

        # get size of dimensions, remembering that x is N-S, y is E-W, z is + down
        nsize = ilines[1].strip().split()
        n_north = int(nsize[0])
        n_east = int(nsize[1])
        n_z = int(nsize[2])
        log_yn = nsize[4]

        # get nodes
        self.nodes_north = np.array(
            [float(nn) for nn in ilines[2].strip().split()]
        )
        self.nodes_east = np.array(
            [float(nn) for nn in ilines[3].strip().split()]
        )
        self.nodes_z = np.array(
            [float(nn) for nn in ilines[4].strip().split()]
        )

        self.res_model = np.zeros((n_north, n_east, n_z))

        # get model
        count_z = 0
        line_index = 6
        count_e = 0
        while count_z < n_z:
            iline = ilines[line_index].strip().split()
            # blank lines spit the depth blocks, use those as a marker to
            # set the layer number and start a new block
            if len(iline) == 0:
                count_z += 1
                count_e = 0
                line_index += 1
            # 3D grid model files don't have a space at the end
            # additional condition to account for this.
            elif (len(iline) == 3) & (count_z == n_z - 1):
                count_z += 1
                count_e = 0
                line_index += 1
            # each line in the block is a line of N-->S values for an east value
            else:
                north_line = np.array([float(nres) for nres in iline])

                # Need to be sure that the resistivity array matches
                # with the grids, such that the first index is the
                # furthest south
                self.res_model[:, count_e, count_z] = north_line[::-1]

                count_e += 1
                line_index += 1

        # --> get grid center and rotation angle
        if len(ilines) > line_index:
            for iline in ilines[line_index:]:
                ilist = iline.strip().split()
                # grid center
                if len(ilist) == 3:
                    self.grid_center = np.array(ilist, dtype=float)
                # rotation angle
                elif len(ilist) == 1:
                    self.mesh_rotation_angle = float(ilist[0])
                else:
                    pass

        # --> make sure the resistivity units are in linear Ohm-m
        if log_yn.lower() == "loge":
            self.res_model = np.e**self.res_model
        elif log_yn.lower() == "log" or log_yn.lower() == "log10":
            self.res_model = 10**self.res_model

        # center the grids
        if self.grid_center is None:
            self.grid_center = np.array(
                [-self.nodes_north.sum() / 2, -self.nodes_east.sum() / 2, 0.0]
            )

        # need to shift the grid if the center is not symmetric
        # use the grid centre from the model file
        shift_north = self.grid_center[0]  # + self.nodes_north.sum() / 2
        shift_east = self.grid_center[1]  # + self.nodes_east.sum() / 2
        shift_z = self.grid_center[2]

        # shift the grid.  if shift is + then that means the center is
        self.grid_north += shift_north
        self.grid_east += shift_east
        self.grid_z += shift_z

        # get cell size
        try:
            self.cell_size_east = stats.mode(self.nodes_east)[0][0]
            self.cell_size_north = stats.mode(self.nodes_north)[0][0]
        except IndexError:
            self.cell_size_east = stats.mode(self.nodes_east).mode
            self.cell_size_north = stats.mode(self.nodes_north).mode

        # get number of padding cells
        half = int(self.nodes_east.size / 2)
        self.pad_east = (
            half
            - np.where(
                (self.nodes_east[0:half] < self.cell_size_east * 1.1)
                & (self.nodes_east[0:half] > self.cell_size_east * 0.9)
            )[0].size
        )
        half = int(self.nodes_north.size / 2)
        self.pad_north = (
            half
            - np.where(
                (self.nodes_north[0:half] < self.cell_size_north * 1.1)
                & (self.nodes_north[0:half] > self.cell_size_north * 0.9)
            )[0].size
        )

        topo = self._get_topography_from_model()
        if topo is not None:
            self.surface_dict["topography"] = topo

        try:
            self.n_air_layers = np.where(self.res_model > 1e10)[-1].max()
        except (IndexError, ValueError):
            self.n_air_layers = 0

        self.n_layers = self.nodes_z.size - self.n_air_layers

    def _get_topography_from_model(self):
        """Get topography from an input model if air layers are found.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        topo = np.zeros((self.res_model.shape[0], self.res_model.shape[1]))
        if np.any(self.res_model[:, :, 0] > 1e7):
            for ii in range(self.res_model.shape[0]):
                for jj in range(self.res_model.shape[1]):
                    try:
                        topo[ii, jj] = (
                            -1
                            * self.grid_z[
                                np.where(self.res_model[ii, jj] > 1e6)[0][-1]
                            ]
                        )
                    except IndexError:
                        topo[ii, jj] = -1 * self.grid_z[0]
            return topo

    def plot_mesh(self, **kwargs):
        """Plot model mesh.
        :param **kwargs:
        :param plot_topography: DESCRIPTION, defaults to False.
        :type plot_topography: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if "topography" in self.surface_dict.keys():
            kwargs["plot_topography"] = True
        return PlotMesh(self, **kwargs)

    @property
    def model_parameters(self):
        """Get important model parameters to write to a file for documentation
        later.
        """

        parameter_list = [
            "cell_size_east",
            "cell_size_north",
            "ew_ext",
            "ns_ext",
            "pad_east",
            "pad_north",
            "pad_z",
            "pad_num",
            "z1_layer",
            "z_target_depth",
            "z_bottom",
            "mesh_rotation_angle",
            "res_initial_value",
            "save_path",
        ]

        parameter_dict = {}
        for parameter in parameter_list:
            key = f"model.{parameter}"
            parameter_dict[key] = getattr(self, parameter)

        parameter_dict["model.size"] = self.res_model.shape

        return parameter_dict

    def to_xarray(self, **kwargs):
        """Put model in xarray format."""

        return xr.DataArray(
            self.res_model,
            coords={
                "north": self.grid_north[0:-1] + self.center_point.north,
                "east": self.grid_east[0:-1] + self.center_point.east,
                "z": self.grid_z[0:-1] + self.center_point.elevation,
            },
            dims=["north", "east", "z"],
            name="resistivity",
            attrs={
                "center_latitude": self.center_point.latitude,
                "center_longitude": self.center_point.longitude,
                "center_elevation": self.center_point.elevation,
                "datum_epsg": self.center_point.datum_epsg,
                "datum_wkt": self.center_point.datum_crs.to_wkt(),
                "center_point_east": self.center_point.east,
                "center_point_north": self.center_point.north,
                "utm_epsg": self.center_point.utm_epsg,
                "utm_wkt": self.center_point.utm_crs.to_wkt(),
            },
        )

    def to_netcdf(self, fn, pad_east=None, pad_north=None, metadata={}):
        """Create a netCDF file to read into GIS software

        works about 50% of the time..
        """
        if self.center_point.utm_epsg is None:
            raise ValueError("Must input UTM CRS or EPSG")

        pad_east = self._validate_pad_east(pad_east)
        pad_north = self._validate_pad_north(pad_north)

        east, north = np.broadcast_arrays(
            self.grid_north[pad_north : -(pad_north + 1), None]
            + self.center_point.north,
            self.grid_east[None, pad_east : -(pad_east + 1)]
            + self.center_point.east,
        )

        lat, lon = project_point(
            north.ravel(),
            east.ravel(),
            self.center_point.utm_epsg,
            self.center_point.datum_epsg,
        )

        latitude = np.linspace(lat.min(), lat.max(), east.shape[0])
        longitude = np.linspace(lon.min(), lon.max(), east.shape[1])
        depth = (self.grid_z[:-1] + self.center_point.elevation) / 1000

        # need to depth, latitude, longitude for NetCDF
        x_res = np.swapaxes(
            np.log10(
                self.res_model[pad_north:-pad_north, pad_east:-pad_east, :]
            ),
            0,
            1,
        ).T
        x = xr.DataArray(
            x_res,
            coords=[
                ("depth", depth),
                ("latitude", latitude),
                ("longitude", longitude),
            ],
            dims=["depth", "latitude", "longitude"],
        )

        # =============================================================================
        # fill in the metadata
        x.name = "electrical_resistivity"
        x.attrs["long_name"] = "electrical resistivity"
        x.attrs["units"] = "Ohm-m"
        x.attrs["standard_name"] = "resistivity"
        x.attrs["display_name"] = "log10(resistivity)"

        # metadata for coordinates
        x.coords["latitude"].attrs["long_name"] = "Latitude; positive_north"
        x.coords["latitude"].attrs["units"] = "degrees_north"
        x.coords["latitude"].attrs["standard_name"] = "Latitude"

        x.coords["longitude"].attrs["long_name"] = "longitude; positive_east"
        x.coords["longitude"].attrs["units"] = "degrees_east"
        x.coords["longitude"].attrs["standard_name"] = "longitude"

        x.coords["depth"].attrs["long_name"] = "depth; positive_down"
        x.coords["depth"].attrs["display_name"] = "depth"
        x.coords["depth"].attrs["units"] = "km"
        x.coords["depth"].attrs["standard_name"] = "depth"

        ds = xr.Dataset(*[{"resistivity": x}])

        # fill in some metadata

        ds.attrs["Conventions"] = "CF-1.0"
        ds.attrs["Metadata_Conventions"] = "Unidata Dataset Discovery v1.0"
        ds.attrs["NCO"] = (
            "netCDF Operators version 4.7.5 (Homepage = http://nco.sf.net, Code=http://github/nco/nco"
        )

        for key, value in metadata.items():
            ds.attrs[key] = value

        # geospatial metadata
        ds.attrs["geospatial_lat_min"] = latitude.min()
        ds.attrs["geospatial_lat_max"] = latitude.max()
        ds.attrs["geospatial_lat_units"] = "degrees_north"
        ds.attrs["geospatial_lat_resolution"] = np.diff(latitude).mean()

        ds.attrs["geospatial_lon_min"] = longitude.min()
        ds.attrs["geospatial_lon_max"] = longitude.max()
        ds.attrs["geospatial_lon_units"] = "degrees_east"
        ds.attrs["geospatial_lon_resolution"] = np.diff(longitude).mean()

        ds.attrs["geospatial_vertical_min"] = depth.min()
        ds.attrs["geospatial_vertical_max"] = depth.max()
        ds.attrs["geospatial_vertical_units"] = "km"
        ds.attrs["geospatial_vertical_positive"] = "down"

        # write to netcdf
        ds.to_netcdf(path=fn)

        return ds

    def to_gocad_sgrid(
        self, fn=None, origin=[0, 0, 0], clip=0, no_data_value=-99999
    ):
        """Write a model to gocad sgrid

        optional inputs:

        fn = filename to save to. File extension ('.sg') will be appended.
             default is the model name with extension removed
        origin = real world [x,y,z] location of zero point in model grid
        clip = how much padding to clip off the edge of the model for export,
               provide one integer value or list of 3 integers for x,y,z directions
        no_data_value = no data value to put in sgrid.
        """
        if not np.iterable(clip):
            clip = [clip, clip, clip]

        # determine save path
        if fn is not None:
            fn = Path(fn)
            # if fn is a full path, convert to a file name
            fndir = fn.parent
            if fndir.is_dir():
                sg_basename = fn.name
            else:
                sg_basename = fn
        else:
            # create a basename if fn is None
            sg_basename = self.model_fn.stem

        self.save_path, fn, sg_basename = mtfh.validate_save_file(
            save_path=self.save_path, savefile=fn, basename=sg_basename
        )

        # number of cells in the ModEM model
        nyin, nxin, nzin = np.array(self.res_model.shape) + 1

        gx, gy = mtmesh.rotate_mesh(
            self.grid_east[clip[0] : nxin - clip[0]],
            self.grid_north[clip[1] : nyin - clip[1]],
            origin[:2],
            self.mesh_rotation_angle,
        )

        gz = -1.0 * self.grid_z[: nzin - clip[2]] - origin[2]

        gxm, gzm = np.meshgrid(gx, gz)
        gym, gzm = np.meshgrid(gy, gz)

        gxm = gxm.reshape(len(gz), len(gy), len(gx[0])).transpose(1, 2, 0)
        gym = gym.reshape(len(gz), len(gy), len(gx[0])).transpose(1, 2, 0)
        gzm = gzm.reshape(len(gz), len(gy), len(gx[0])).transpose(1, 2, 0)

        gridedges = (gxm, gym, gzm)

        # resistivity values, clipped to one smaller than grid edges
        resvals = self.res_model[
            clip[1] : nyin - clip[1] - 1,
            clip[0] : nxin - clip[0] - 1,
            : nzin - clip[2] - 1,
        ]

        sg_obj = mtgocad.Sgrid(
            resistivity=resvals,
            grid_xyz=gridedges,
            fn=sg_basename,
            workdir=self.save_path,
        )
        sg_obj.write_sgrid_file()

    def from_gocad_sgrid(
        self,
        sgrid_header_file,
        air_resistivity=1e39,
        sea_resistivity=0.3,
        sgrid_positive_up=True,
    ):
        """Read a gocad sgrid file and put this info into a ModEM file.

        Note: can only deal with grids oriented N-S or E-W at this stage,
        with orthogonal coordinates
        """
        # read sgrid file
        sg_obj = mtgocad.Sgrid()
        sg_obj.read_sgrid_file(sgrid_header_file)
        self.sg_obj = sg_obj

        # get resistivity model values
        self.res_model = sg_obj.resistivity

        # get nodes and grid locations
        grideast, gridnorth, gridz = [
            np.unique(sg_obj.grid_xyz[i]) for i in range(3)
        ]
        # check if sgrid is positive up and convert to positive down if it is
        # (ModEM grid is positive down)
        if sgrid_positive_up:
            gridz = -gridz

        gridz.sort()

        if np.all(
            np.array([len(gridnorth), len(grideast), len(gridz)]) - 1
            == np.array(self.res_model.shape)
        ):
            self.grid_east, self.grid_north, self.grid_z = (
                grideast,
                gridnorth,
                gridz,
            )
        else:
            print(
                "Cannot read sgrid, can't deal with non-orthogonal grids or grids not aligned N-S or E-W"
            )
            return

        # check if we have a data object and if we do, is there a centre position
        # if not then assume it is the centre of the grid
        calculate_centre = True
        if self.data_obj is not None:
            if hasattr(self.data_obj, "center_point"):
                if self.data_obj.center_point is not None:
                    centre = np.zeros(3)
                    centre[0] = self.data_obj.center_point["east"]
                    centre[1] = self.data_obj.center_point["north"]
                    calculate_centre = False
        # get relative grid locations
        if calculate_centre:
            print("Calculating center position")
            centre = np.zeros(3)
            centre[0] = (self.grid_east.max() + self.grid_east.min()) / 2.0
            centre[1] = (self.grid_north.max() + self.grid_north.min()) / 2.0
        centre[2] = self.grid_z[0]

        self.grid_east -= centre[0]
        self.grid_north -= centre[1]

        self.grid_center = np.array(
            [self.grid_north[0], self.grid_east[0], self.grid_z[0]]
        )

        self.z1_layer = self.nodes_z[0]
        #        self.z_target_depth = None
        self.z_bottom = self.nodes_z[-1]

        # number of vertical layers
        self.n_layers = len(self.grid_z) - 1

        # number of air layers
        self.n_airlayers = sum(
            np.amax(self.res_model, axis=(0, 1)) > 0.9 * air_resistivity
        )

        # sea level in grid_z coordinates, calculate and adjust centre
        self.sea_level = self.grid_z[self.n_airlayers]

    def interpolate_elevation(
        self,
        surface_file=None,
        surface=None,
        get_surface_name=False,
        method="nearest",
        fast=True,
        shift_north=0,
        shift_east=0,
    ):
        """Project a surface to the model grid and add resulting elevation data
        to a dictionary called surface_dict. Assumes the surface is in lat/long
        coordinates (wgs84)

        **returns**
        nothing returned, but surface data are added to surface_dict under
        the key given by surface_name.

        **inputs**
        choose to provide either surface_file (path to file) or surface (tuple).
        If both are provided then surface tuple takes priority.

        surface elevations are positive up, and relative to sea level.
        surface file format is:

        ncols         3601
        nrows         3601
        xllcorner     -119.00013888889 (longitude of lower left)
        yllcorner     36.999861111111  (latitude of lower left)
        cellsize      0.00027777777777778
        NODATA_value  -9999
        elevation data W --> E
        N
        |
        V
        S

        Alternatively, provide a tuple with:
        (lon,lat,elevation)
        where elevation is a 2D array (shape (ny,nx)) containing elevation
        points (order S -> N, W -> E)
        and lon, lat are either 1D arrays containing list of longitudes and
        latitudes (in the case of a regular grid) or 2D arrays with same shape
        as elevation array containing longitude and latitude of each point.

        other inputs:
        surface_epsg = epsg number of input surface, default is 4326 for lat/lon(wgs84)
        method = interpolation method. Default is 'nearest', if model grid is
        dense compared to surface points then choose 'linear' or 'cubic'
        """
        # initialise a dictionary to contain the surfaces
        if not hasattr(self, "surface_dict"):
            self.surface_dict = {}

        # get centre position of model grid in real world coordinates
        x0, y0 = (
            self.center_point.east + shift_east,
            self.center_point.north + shift_north,
        )

        if self.mesh_rotation_angle is None:
            self.mesh_rotation_angle = 0

        xg, yg = mtmesh.rotate_mesh(
            self.grid_east,
            self.grid_north,
            [x0, y0],
            self.mesh_rotation_angle,
            return_centre=True,
        )
        if surface_file:
            elev_mg = mtmesh.interpolate_elevation_to_grid(
                xg,
                yg,
                surface_file=surface_file,
                utm_epsg=self.model_epsg,
                datum_epsg=self.center_point.datum_epsg,
                method=method,
                fast=fast,
            )
        elif surface:
            # Always use fast=False when reading from EDI data because
            #  we're already providing a subset of the grid.
            elev_mg = mtmesh.interpolate_elevation_to_grid(
                xg,
                yg,
                surface=surface,
                utm_epsg=self.model_epsg,
                datum_epsg=self.center_point.datum_epsg,
                method=method,
                fast=False,
            )
        else:
            raise ValueError("'surface_file' or 'surface' must be provided")

        # get a name for surface
        if get_surface_name:
            if surface_file is not None:
                surface_file = Path(surface_file)
                surface_name = surface_file.name
            else:
                ii = 1
                surface_name = f"surface{int(ii):01}"
                while surface_name in list(self.surface_dict.keys()):
                    ii += 1
                    surface_name = f"surface{int(ii):01}"
            return elev_mg, surface_name
        else:
            return elev_mg

    def add_topography_from_data(
        self,
        interp_method="nearest",
        air_resistivity=1e12,
        topography_buffer=None,
        airlayer_type="log_up",
    ):
        """Wrapper around add_topography_to_model that allows creating
        a surface model from EDI data. The Data grid and station
        elevations will be used to make a 'surface' tuple that will
        be passed to add_topography_to_model so a surface model
        can be interpolated from it.

        The surface tuple is of format (lon, lat, elev) containing
        station locations.
        :param data_object: A ModEm data
            object that has been filled with data from EDI files.
        :type data_object: mtpy.modeling.ModEM.data.Data
        :param interp_method: Same as
            add_topography_to_model, defaults to "nearest".
        :type interp_method: str, optional
        :param air_resistivity: Same as
            add_topography_to_model, defaults to 1e12.
        :type air_resistivity: float, optional
        :param topography_buffer: Same as
            add_topography_to_model, defaults to None.
        :type topography_buffer: float, optional
        :param airlayer_type: Same as
            add_topography_to_model, defaults to "log_up".
        :type airlayer_type: str, optional
        """
        lon = self.station_locations.longitude.to_numpy()
        lat = self.station_locations.latitude.to_numpy()
        elev = self.station_locations.elevation.to_numpy()
        surface = lon, lat, elev
        self.add_topography_to_model(
            surface=surface,
            interp_method=interp_method,
            air_resistivity=air_resistivity,
            topography_buffer=topography_buffer,
            airlayer_type=airlayer_type,
        )

    def add_topography_to_model(
        self,
        topography_file=None,
        surface=None,
        topography_array=None,
        interp_method="nearest",
        air_resistivity=1e12,
        topography_buffer=None,
        airlayer_type="log_up",
        max_elev=None,
        shift_east=0,
        shift_north=0,
    ):
        """If air_layers is non-zero, will add topo: read in topograph file,
        make a surface model.

        Call project_stations_on_topography in the end, which will re-write
        the .dat file.

        If n_airlayers is zero, then cannot add topo data, only bathymetry is needed.
        :param shift_north:
            Defaults to 0.
        :param shift_east:
            Defaults to 0.
        :param max_elev:
            Defaults to None.
        :param surface:
            Defaults to None.
        :param topography_file: File containing topography (arcgis ascii grid), defaults to None.
        :param topography_array: Alternative to topography_file - array of
            elevation values on model grid, defaults to None.
        :param interp_method: Interpolation method for topography,
            'nearest', 'linear', or 'cubic', defaults to "nearest".
        :param air_resistivity: Resistivity value to assign to air, defaults to 1e12.
        :param topography_buffer: Buffer around stations to calculate minimum
            and maximum topography value to use for
            meshing, defaults to None.
        :param airlayer_type: How to set air layer thickness - options are
            'constant' for constant air layer thickness,
            or 'log', for logarithmically increasing air
            layer thickness upward, defaults to "log_up".
        """
        # first, get surface data
        if topography_file:
            self.surface_dict["topography"] = self.interpolate_elevation(
                surface_file=topography_file,
                method=interp_method,
                shift_east=shift_east,
                shift_north=shift_north,
            )
        elif surface:
            self.surface_dict["topography"] = self.interpolate_elevation(
                surface=surface,
                method=interp_method,
                shift_east=shift_east,
                shift_north=shift_north,
            )
        elif topography_array:
            self.surface_dict["topography"] = topography_array
        else:
            raise ValueError(
                "'topography_file', 'surface' or "
                + "topography_array must be provided"
            )

        if self.n_air_layers is None or self.n_air_layers == 0:
            self._logger.warning(
                "No air layers specified, so will not add air/topography !!!"
            )
            self._logger.warning(
                "Only bathymetry will be added below according to the topofile: sea-water low resistivity!!!"
            )

        elif (
            self.n_air_layers > 0
        ):  # FZ: new logic, add equal blocksize air layers on top of the simple flat-earth grid
            # get grid centre
            gcx, gcy = [
                np.mean([arr[:-1], arr[1:]], axis=0)
                for arr in (self.grid_east, self.grid_north)
            ]
            # get core cells
            if topography_buffer is None:
                topography_buffer = (
                    5
                    * (self.cell_size_east**2 + self.cell_size_north**2) ** 0.5
                )
            core_cells = mtmesh.get_station_buffer(
                gcx,
                gcy,
                self.station_locations["model_east"],
                self.station_locations["model_north"],
                buf=topography_buffer,
            )
            topo_core = self.surface_dict["topography"][core_cells]
            topo_core_min = max(topo_core.min(), 0)

            if airlayer_type == "log_up":
                # log increasing airlayers, in reversed order
                new_air_nodes = mtmesh.make_log_increasing_array(
                    self.z1_layer,
                    topo_core.max() - topo_core_min,
                    self.n_air_layers,
                    increment_factor=0.999,
                )[::-1]
            elif airlayer_type == "log_down":
                # make a new mesh
                n_layers = self.n_layers + self.n_air_layers
                self.nodes_z, z_grid = self.make_z_mesh(n_layers)

                # adjust level to topography min
                if max_elev is not None:
                    self.grid_z -= max_elev
                    ztops = np.where(
                        self.surface_dict["topography"] > max_elev
                    )
                    self.surface_dict["topography"][ztops] = max_elev
                else:
                    self.grid_z -= topo_core.max()

            elif airlayer_type == "constant":
                if max_elev is not None:
                    air_cell_thickness = np.ceil(
                        (max_elev - topo_core_min) / self.n_air_layers
                    )
                else:
                    air_cell_thickness = np.ceil(
                        (topo_core.max() - topo_core_min) / self.n_air_layers
                    )
                new_air_nodes = np.array(
                    [air_cell_thickness] * self.n_air_layers
                )

            if "down" not in airlayer_type:
                # sum to get grid cell locations
                new_airlayers = np.array(
                    [
                        new_air_nodes[:ii].sum()
                        for ii in range(len(new_air_nodes) + 1)
                    ]
                )
                # maximum topography cell on the grid
                topo_max_grid = topo_core_min + new_airlayers[-1]
                # round to nearest whole number and convert subtract the max elevation (so that sea level is at topo_core_min)
                new_airlayers = np.around(new_airlayers - topo_max_grid)
                # add new air layers, cut_off some tailing layers to preserve array size.
                self.grid_z = np.concatenate(
                    [new_airlayers[:-1], self.grid_z + new_airlayers[-1]],
                    axis=0,
                )

            self._logger.debug(f"self.grid_z[0:2] {self.grid_z[0:2]}")

        # update the z-centre as the top air layer
        self.grid_center[2] = self.grid_z[0]

        # update the resistivity model
        new_res_model = (
            np.ones(
                (
                    self.nodes_north.size,
                    self.nodes_east.size,
                    self.nodes_z.size,
                )
            )
            * self.res_initial_value
        )

        if "down" not in airlayer_type:
            new_res_model[:, :, self.n_air_layers :] = self.res_model

        self.res_model = new_res_model

        # assign topography
        top = np.zeros_like(self.surface_dict["topography"]) + self.grid_z[0]
        bottom = -self.surface_dict["topography"]
        self.assign_resistivity_from_surface_data(top, bottom, air_resistivity)
        # assign bathymetry
        self.assign_resistivity_from_surface_data(
            np.zeros_like(top), bottom, 0.3
        )

        return

    def _validate_extent(self, east, west, south, north, extent_ratio=2.0):
        """Validate the provided ew_ext and ns_ext to make sure the model fits
        within these extents and allows enough space for padding according to
        the extent ratio provided. If not, then update ew_ext and ns_ext parameters
        """
        inner_ew_ext = west - east
        inner_ns_ext = north - south

        if self.ew_ext < extent_ratio * inner_ew_ext:
            self._logger.warning(
                "Provided or default ew_ext not sufficient to fit stations + padding, updating extent"
            )
            self.ew_ext = np.ceil(extent_ratio * inner_ew_ext)

        if self.ns_ext < extent_ratio * inner_ns_ext:
            self._logger.warning(
                "Provided or default ns_ext not sufficient to fit stations + padding, updating extent"
            )
            self.ns_ext = np.ceil(extent_ratio * inner_ns_ext)

    def interpolate_to_even_grid(
        self, cell_size, pad_north=None, pad_east=None
    ):
        """Interpolate the model onto an even grid for plotting as a raster or
        netCDF.
        :param cell_size: DESCRIPTION.
        :type cell_size: TYPE
        :param pad_north: DESCRIPTION, defaults to None.
        :type pad_north: TYPE, optional
        :param pad_east: DESCRIPTION, defaults to None.
        :type pad_east: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        pad_east = self._validate_pad_east(pad_east)
        pad_north = self._validate_pad_north(pad_north)

        # need -2 because grid is + 1 of size
        new_east = np.arange(
            self.grid_east[pad_east],
            self.grid_east[-pad_east - 2],
            cell_size,
        )
        new_north = np.arange(
            self.grid_north[pad_north],
            self.grid_north[-pad_north - 2],
            cell_size,
        )

        # needs to be -1 because the grid is n+1 as it is the edges of the
        # the nodes.
        model_n, model_e = np.broadcast_arrays(
            self.grid_north[:-1, None],
            self.grid_east[None, :-1],
        )

        new_res_arr = np.zeros(
            (new_north.size, new_east.size, self.nodes_z.size)
        )

        for z_index in range(self.grid_z.shape[0] - 1):
            res = self.res_model[:, :, z_index]
            new_res_arr[:, :, z_index] = interpolate.griddata(
                (model_n.ravel(), model_e.ravel()),
                res.ravel(),
                (new_north[:, None], new_east[None, :]),
            )

        return new_north, new_east, new_res_arr

    def get_lower_left_corner(
        self, pad_east, pad_north, shift_east=0, shift_north=0
    ):
        """Get the lower left corner in UTM coordinates for raster.
        :param shift_north:
            Defaults to 0.
        :param shift_east:
            Defaults to 0.
        :param pad_east: Number of padding cells to skip from outside in.
        :type pad_east: integer
        :param pad_north: Number of padding cells to skip from outside in.
        :type pad_north: integer
        :return: Lower left hand corner.
        :rtype: :class:`mtpy.core.MTLocation`
        """

        if self.center_point.utm_crs is None:
            raise ValueError("Need to input center point and UTM CRS.")

        lower_left = MTLocation()
        lower_left.utm_crs = self.center_point.utm_crs
        lower_left.datum_crs = self.center_point.datum_crs
        lower_left.east = (
            self.center_point.east + self.grid_east[pad_east] + shift_east
        )
        lower_left.north = (
            self.center_point.north + self.grid_north[pad_north] + shift_north
        )

        return lower_left

    def _get_depth_min_index(self, depth_min):
        """Get index of minimum depth, if None, return None.
        :param depth_min: DESCRIPTION.
        :type depth_min: TYPE
        :raises IndexError: DESCRIPTION.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if depth_min is not None:
            try:
                depth_min = np.where(self.grid_z >= depth_min)[0][0]
            except IndexError:
                raise IndexError(
                    f"Could not locate depths deeper than {depth_min}."
                )
        return depth_min

    def _get_depth_max_index(self, depth_max):
        """Get index of minimum depth, if None, return None.
        :param depth_max: DESCRIPTION.
        :type depth_max: TYPE
        :raises IndexError: DESCRIPTION.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if depth_max is not None:
            try:
                depth_max = np.where(self.grid_z <= depth_max)[0][-1]
            except IndexError:
                raise IndexError(
                    f"Could not locate depths shallower than {depth_max}."
                )
        return depth_max

    def _get_pad_slice(self, pad):
        """Get padding slice.
        :param pad: DESCRIPTION.
        :type pad: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if pad in [None, 0]:
            return slice(None, None)
        else:
            return slice(pad, -pad)

    def _validate_pad_east(self, pad_east):
        """Pad east if None, return self.pad_east."""
        if pad_east is None:
            return self.pad_east
        elif pad_east == 0:
            return None
        return pad_east

    def _validate_pad_north(self, pad_north):
        """Pad north if None, return self.pad_north."""
        if pad_north is None:
            return self.pad_north
        return pad_north

    def _validate_pad_z(self, pad_z):
        """Validate pad north."""

        if pad_z is None:
            return self.pad_z
        elif pad_z == 0:
            return None
        return pad_z

    def _clip_model(self, pad_east, pad_north, pad_z):
        """Clip model based on excluding the number of padding cells.
        :param pad_z:
        :param pad_east: DESCRIPTION.
        :type pad_east: TYPE
        :param pad_north: DESCRIPTION.
        :type pad_north: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        return self.res_model[
            self._get_pad_slice(pad_north),
            self._get_pad_slice(pad_east),
            0:pad_z,
        ]

    def to_raster(
        self,
        cell_size,
        pad_north=None,
        pad_east=None,
        save_path=None,
        depth_min=None,
        depth_max=None,
        rotation_angle=0,
        shift_north=0,
        shift_east=0,
        log10=True,
        verbose=True,
    ):
        """Write out each depth slice as a raster in UTM coordinates.

                Expecting
        a grid that is interoplated onto a regular grid of square cells with
                size `cell_size`.
                :param verbose:
                    Defaults to True.
                :param log10:
                    Defaults to True.
                :param shift_east:
                    Defaults to 0.
                :param shift_north:
                    Defaults to 0.
                :param cell_size: Square cell size (cell_size x cell_size) in meters.
                :type cell_size: float
                :param pad_north: Number of padding cells to skip from outside in,
                    if None defaults to self.pad_north, defaults to None.
                :type pad_north: integer, optional
                :param pad_east: Number of padding cells to skip from outside in
                    if None defaults to self.pad_east, defaults to None.
                :type pad_east: integer, optional
                :param save_path: Path to save files to. If None use self.save_path,, defaults to None.
                :type save_path: string or Path, optional
                :param depth_min: Minimum depth to make raster for in meters,, defaults to None.
                :type depth_min: float, optional
                :param depth_max: Maximum depth to make raster for in meters,, defaults to None.
                :type depth_max: float, optional
                :param rotation_angle: Angle (degrees) to rotate the raster assuming
                    clockwise positive rotation where North = 0, East = 90, defaults to 0.
                :type rotation_angle: float, optional
                :raises ValueError: If utm_epsg is not input.
                :return: List of file paths to rasters.
                :rtype: TYPE
        """

        if self.center_point.utm_crs is None:
            raise ValueError("Need to input center point and UTM CRS.")

        if rotation_angle is not None:
            rotation_angle = float(rotation_angle)
        else:
            rotation_angle = 0.0

        if save_path is None:
            save_path = self.save_path
        else:
            save_path = Path(save_path)

        if not save_path.exists():
            save_path.mkdir()

        pad_east = self._validate_pad_east(pad_east)
        pad_north = self._validate_pad_north(pad_north)

        _, _, raster_array = self.interpolate_to_even_grid(
            cell_size, pad_east=pad_east, pad_north=pad_north
        )

        if log10:
            raster_array = np.log10(raster_array)

        raster_depths = self.grid_z[
            slice(
                self._get_depth_min_index(depth_min),
                self._get_depth_max_index(depth_max),
            )
        ]

        initial_index = np.where(self.grid_z == raster_depths.min())[0][0]

        lower_left = self.get_lower_left_corner(
            pad_east, pad_north, shift_east=shift_east, shift_north=shift_north
        )

        raster_fn_list = []
        for ii, d in enumerate(raster_depths, initial_index):
            try:
                raster_fn = save_path.joinpath(
                    f"{ii:02}_depth_{d:.2f}m_utm_{self.center_point.utm_epsg}.tif".replace(
                        "-", "m"
                    )
                )
                array2raster(
                    raster_fn,
                    raster_array[:, :, ii],
                    lower_left,
                    cell_size,
                    cell_size,
                    self.center_point.utm_epsg,
                    rotation_angle=rotation_angle,
                )
                raster_fn_list.append(raster_fn)
                if verbose:
                    self._logger.info(f"Wrote depth index {ii} to {raster_fn}")
            except IndexError:
                break

        return raster_fn_list

    def to_conductance_raster(
        self,
        cell_size,
        conductance_dict,
        pad_north=None,
        pad_east=None,
        save_path=None,
        rotation_angle=0,
        shift_north=0,
        shift_east=0,
        log10=True,
        verbose=True,
    ):
        """Write out a raster in UTM coordinates for conductance sections.

        Expecting a grid that is interoplated onto a regular grid of square
        cells with size `cell_size`.

        `conductance_dict = {"layer_name": (min_depth, max_depth)}

        if "layer_name" is "surface" then topography is included
        :param verbose:
            Defaults to True.
        :param log10:
            Defaults to True.
        :param shift_east:
            Defaults to 0.
        :param shift_north:
            Defaults to 0.
        :param cell_size: Square cell size (cell_size x cell_size) in meters.
        :type cell_size: float
        :param conductance_dict: DESCRIPTION.
        :type conductance_dict: TYPE
        :param pad_north: Number of padding cells to skip from outside in,
            if None defaults to self.pad_north, defaults to None.
        :type pad_north: integer, optional
        :param pad_east: Number of padding cells to skip from outside in
            if None defaults to self.pad_east, defaults to None.
        :type pad_east: integer, optional
        :param save_path: Path to save files to. If None use self.save_path,, defaults to None.
        :type save_path: string or Path, optional
        :param depth_min: Minimum depth to make raster for in meters,, defaults to None which will use shallowest depth.
        :type depth_min: float, optional
        :param depth_max: Maximum depth to make raster for in meters,, defaults to None which will use deepest depth.
        :type depth_max: float, optional
        :param rotation_angle: Angle (degrees) to rotate the raster assuming
            clockwise positive rotation where North = 0, East = 90, defaults to 0.
        :type rotation_angle: float, optional
        :raises ValueError: If utm_epsg is not input.
        :return: List of file paths to rasters.
        :rtype: TYPE
        """

        if self.center_point.utm_crs is None:
            raise ValueError("Need to input center point and UTM CRS.")

        if rotation_angle is not None:
            rotation_angle = float(rotation_angle)
        else:
            rotation_angle = 0.0

        if save_path is None:
            save_path = self.save_path
        else:
            save_path = Path(save_path)

        if not save_path.exists():
            save_path.mkdir()

        pad_east = self._validate_pad_east(pad_east)
        pad_north = self._validate_pad_north(pad_north)

        _, _, raster_array = self.interpolate_to_even_grid(
            cell_size, pad_east=pad_east, pad_north=pad_north
        )

        raster_array[np.where(raster_array > 1e10)] = np.nan

        lower_left = self.get_lower_left_corner(
            pad_east, pad_north, shift_east=shift_east, shift_north=shift_north
        )

        raster_fn_list = []
        for ii, key in enumerate(conductance_dict.keys()):
            z = np.array(conductance_dict[key])
            if key in ["surface"]:
                index_min = 0
            else:
                index_min = np.where(self.grid_z <= z.min())[0][-1]
            index_max = np.where(self.grid_z >= z.max())[0][0]

            conductance = (
                1.0 / raster_array[:, :, index_min:index_max]
            ) * abs(self.grid_z[index_min:index_max])
            conductance = np.log10(np.nansum(conductance, axis=2))
            try:
                raster_fn = self.save_path.joinpath(
                    f"conductance_{z.min()}m_to_{z.max()}m_depth_utm_{self.center_point.utm_epsg}.tif".replace(
                        "-", "m"
                    )
                )
                array2raster(
                    raster_fn,
                    conductance,
                    lower_left,
                    cell_size,
                    cell_size,
                    self.center_point.utm_epsg,
                    rotation_angle=rotation_angle,
                )
                raster_fn_list.append(raster_fn)
                if verbose:
                    self._logger.info(
                        f"Wrote conductance {key} to {raster_fn}"
                    )
            except IndexError:
                break

        return raster_fn_list

    def _get_xyzres(self, location_type, pad_east=None, pad_north=None):
        """Get xyz resistivity.
        :param pad_north:
            Defaults to None.
        :param pad_east:
            Defaults to None.
        :param location_type: 'll' or 'utm'.
        :type location_type: TYPE
        :param origin: DESCRIPTION.
        :type origin: TYPE
        :param model_epsg: DESCRIPTION.
        :type model_epsg: TYPE
        :param clip: DESCRIPTION.
        :type clip: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        # reshape the data and get grid centres
        x, y, z = [
            np.mean([arr[1:], arr[:-1]], axis=0)
            for arr in [
                self.grid_east + self.center_point.east,
                self.grid_north + self.center_point.north,
                self.grid_z,
            ]
        ]

        pad_east = self._validate_pad_east(pad_east)
        pad_north = self._validate_pad_north(pad_north)

        x, y, z = np.meshgrid(
            x[slice(pad_east, -pad_east), slice(pad_north, -pad_north)], z
        )

        # set format for saving data
        fmt = ["%.1f", "%.1f", "%.3e"]

        # convert to lat/long if needed
        if location_type in ["ll"]:
            xp, yp = project_point(x, y, self.center_point.utm_epsg, 4326)

            # update format to accommodate lat/lon
            fmt[:2] = ["%.6f", "%.6f"]
        else:
            xp, yp = x, y

        resvals = self.res_model[
            slice(pad_north, -pad_north), slice(pad_east, -pad_east)
        ]

        return xp, yp, z, resvals, fmt

    def to_xyzres(
        self,
        savefile=None,
        location_type="EN",
        log_res=False,
        pad_east=None,
        pad_north=None,
    ):
        """Save a model file as a space delimited x y z res file."""
        xp, yp, z, resvals, fmt = self._get_xyzres(
            location_type, pad_east=pad_east, pad_north=pad_north
        )
        fmt.insert(2, "%.1f")
        xp, yp, z, resvals = (
            xp.flatten(),
            yp.flatten(),
            z.flatten(),
            resvals.flatten(),
        )

        np.savetxt(savefile, np.vstack([xp, yp, z, resvals]).T, fmt=fmt)

    def to_xyres(
        self,
        save_path=None,
        location_type="EN",
        depth_index="all",
        outfile_basename="DepthSlice",
        log_res=False,
        pad_east=None,
        pad_north=None,
    ):
        """Write files containing depth slice data (x, y, res for each depth)

        origin = x,y coordinate of zero point of ModEM_grid, or name of file
                 containing this info (full path or relative to model files)
        save_path = path to save to, default is the model object save path
        location_type = 'EN' or 'LL' xy points saved as eastings/northings or
                        longitude/latitude, if 'LL' need to also provide model_epsg
        model_epsg = epsg number that was used to project the model
        outfile_basename = string for basename for saving the depth slices.
        log_res = True/False - option to save resistivity values as log10
                               instead of linear
        clip = number of cells to clip on each of the east/west and north/south edges.
        """
        if save_path is None:
            save_path = Path(self.save_path)
        else:
            save_path = Path(save_path)
        # make a directory to save the files
        save_path = save_path.joinpath(outfile_basename)
        if not save_path.exists():
            save_path.mkdir()

        xp, yp, z, resvals, fmt = self._get_xyzres(
            location_type, pad_east=pad_east, pad_north=pad_north
        )
        xp = xp[:, :, 0].flatten()
        yp = yp[:, :, 0].flatten()

        # make depth indices into a list
        if depth_index == "all":
            depthindices = list(range(z.shape[2]))
        elif np.iterable(depth_index):
            depthindices = np.array(depth_index).astype(int)
        else:
            depthindices = [depth_index]

        for k in depthindices:
            fname = save_path.joinpath(
                outfile_basename + f"_{int(self.grid_z[k]):1}m.xyz"
            )

            # get relevant depth slice
            vals = resvals[:, :, k].flatten()

            if log_res:
                vals = np.log10(vals)
                fmt[-1] = "%.3f"
            data = np.vstack([xp, yp, vals]).T

            np.savetxt(fname, data, fmt=fmt)

    def _rotate_res_model(self):
        """Rotate `res_model` for some reason when you do rot90 the flags of
        the numpy array get set to False and causes an error in pyevtk.  Need
        a little trick to keep the C_CONTIGUOUS = True.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        rotated = np.swapaxes(self.res_model, 0, 1)

        return rotated.copy()

    def to_vtk(
        self,
        vtk_fn=None,
        vtk_save_path=None,
        vtk_fn_basename="ModEM_model_res",
        **kwargs,
    ):
        """Write a VTK file to plot in 3D rendering programs like Paraview.
        :param **kwargs:
        :param vtk_fn: Full path to VKT file to be written, defaults to None.
        :type vtk_fn: string or Path, optional
        :param vtk_save_path: Directory to save vtk file to, defaults to None.
        :type vtk_save_path: string or Path, optional
        :param vtk_fn_basename: Filename basename of vtk file, note that .vtr
            extension is automatically added, defaults to "ModEM_model_res".
        :type vtk_fn_basename: string, optional
        :param geographic_coordinates: [ True | False ] True for geographic
            coordinates.
        :type geographic_coordinates: boolean, optional
        :param units: Units of the spatial grid [ km | m | ft ], defaults to "km".
        :type units: string, optional
        :type: string
        :param coordinate_system: Coordinate system for the station, either the
            normal MT right-hand coordinate system with z+ down or the sinister
            z- down [ nez+ | enz- ], defaults to nez+.
        :return: Full path to VTK file.
        :rtype: Path
        """

        if vtk_fn is None:
            if vtk_save_path is None:
                raise ValueError("Need to input vtk_save_path")
            vtk_fn = Path(vtk_save_path, vtk_fn_basename)
        else:
            vtk_fn = Path(vtk_fn)

        if vtk_fn.suffix != "":
            vtk_fn = vtk_fn.parent.joinpath(vtk_fn.stem)

        vtk_x, vtk_y, vtk_z, resistivity = self._to_output_coordinates(
            **kwargs
        )

        label = kwargs.get("model_units", "resistivity")
        cell_data = {label: resistivity}

        gridToVTK(vtk_fn.as_posix(), vtk_x, vtk_y, vtk_z, cellData=cell_data)

        self._logger.info(f"Wrote VTK model file to {vtk_fn}")

    def _to_output_coordinates(
        self,
        geographic=False,
        units="km",
        coordinate_system="nez+",
        output_epsg=None,
        pad_east=0,
        pad_north=0,
        pad_z=0,
        shift_east=0,
        shift_north=0,
        shift_z=0,
        model_units="resistivity",
    ):
        """Create x, y, z, res outputs in requested coordinate system and units

        Parameters are.
        :param geographic: [ True | False ] true for output in geographic
            coordinates, False for relative model coordinates, defaults to False.
        :type geographic: bool, optional
        :param units: [ 'm' | 'km' | 'ft' ], defaults to "km".
        :type units: str, optional
        :param coordinate_system: [ 'nez+' | 'enz-'], defaults to "nez+".
        :type coordinate_system: str, optional
        :param output_epsg: Output EPSG number, if None uses
            center_point.utm_epsg if geographic is True, defaults to None.
        :type output_epsg: int, optional
        :param pad_east: Number of cells to discard on each side in the east,, defaults to 0.
        :type pad_east: int, optional
        :param pad_north: Number of cells to discard on each side in the east,, defaults to 0.
        :type pad_north: int, optional
        :param pad_z: Number of cells to discard at bottom of model,, defaults to 0.
        :type pad_z: int, optional
        :param shift_east: Shift model east [in units], defaults to 0.
        :type shift_east: float, optional
        :param shift_north: Shift model north [in units], defaults to 0.
        :type shift_north: float, optional
        :param shift_z: Shift model vertically [in units], defaults to 0.
        :type shift_z: float, optional
        :param model_units: ["resistivity" | "conductivity" ],, defaults to "resistivity".
        :type model_units: string, optional
        :return: X, y, z, res.
        :rtype: float
        """

        if isinstance(units, str):
            if units.lower() == "km":
                scale = 1.0 / 1000.00
            elif units.lower() == "m":
                scale = 1.0
            elif units.lower() == "ft":
                scale = 3.2808
        elif isinstance(units, (int, float)):
            scale = units

        pad_z = self._validate_pad_z(pad_z)

        east_slice = self._get_pad_slice(self._validate_pad_east(pad_east))
        north_slice = self._get_pad_slice(self._validate_pad_north(pad_north))
        if pad_z is None:
            z_slice = slice(0, None)
        else:
            z_slice = slice(0, -pad_z)

        if output_epsg is not None:
            cp = self.center_point.copy()
            cp.utm_epsg = output_epsg
        else:
            cp = self.center_point

        if geographic:
            shift_north = self.center_point.north
            shift_east = self.center_point.east
            if self.grid_z[0] == self.center_point.elevation:
                shift_z = 0
            else:
                shift_z = self.center_point.elevation

            if "+" in coordinate_system:
                y = (self.grid_east[east_slice] + shift_east) * scale
                x = (self.grid_north[north_slice] + shift_north) * scale
                depth = (self.grid_z[z_slice] + shift_z) * scale
            elif "-" in coordinate_system:
                x = (self.grid_east[east_slice] + shift_east) * scale
                y = (self.grid_north[north_slice] + shift_north) * scale
                depth = -1 * (self.grid_z[z_slice] - shift_z) * scale

            resistivity = self._rotate_res_model()[
                east_slice, north_slice, z_slice
            ]

        # use cellData, this makes the grid properly as grid is n+1
        else:
            if coordinate_system == "nez+":
                x = (self.grid_north[north_slice] + shift_north) * scale
                y = (self.grid_east[east_slice] + shift_east) * scale
                depth = (self.grid_z[z_slice] + shift_z) * scale
                resistivity = self._clip_model(pad_east, pad_north, pad_z)

            elif coordinate_system == "enz-":
                y = (self.grid_north[north_slice] + shift_north) * scale
                x = (self.grid_east[east_slice] + shift_east) * scale
                depth = -1 * (self.grid_z[z_slice] - shift_z) * scale
                resistivity = self._rotate_res_model()[
                    east_slice, north_slice, z_slice
                ]

        return x, y, depth, resistivity

    def to_geosoft_xyz(
        self,
        save_fn,
        pad_north=0,
        pad_east=0,
        pad_z=0,
    ):
        """Write an XYZ file readable by Geosoft

        All input units are in meters..
        :param save_fn: Full path to save file to.
        :type save_fn: string or Path
        :param pad_north: Number of cells to cut from the north-south edges, defaults to 0.
        :type pad_north: int, optional
        :param pad_east: Number of cells to cut from the east-west edges, defaults to 0.
        :type pad_east: int, optional
        :param pad_z: Number of cells to cut from the bottom, defaults to 0.
        :type pad_z: int, optional
        """
        lines = [
            r"/ ------------------------------------------------------------------------------",
            r"/ XYZ  IMPORT [01/25/2021]",
            r"/ VOXEL   [.\electrical_resistivity.geosoft_voxel]",
            r"/ ------------------------------------------------------------------------------",
            r"/ X,Y,Z,Data",
        ]

        # --> write model xyz file
        for kk, zz in enumerate(self.grid_z[0:-pad_z]):
            for jj, yy in enumerate(self.grid_east[pad_east:-pad_east]):
                for ii, xx in enumerate(self.grid_north[pad_north:-pad_north]):
                    lines.append(
                        f"{yy + self.center_point.east:.3f} "
                        f"{xx + self.center_point.north:.3f} "
                        f"{-(zz + self.center_point.elevation):.3f} "
                        f"{self.res_model[ii, jj, kk]:.3f}"
                    )

        with open(save_fn, "w") as fid:
            fid.write("\n".join(lines))

    def to_winglink_out(
        self,
        save_fn,
    ):
        """Will write an .out file for LeapFrog.

        Note that y is assumed to be S --> N, e is assumed to be W --> E and
        z is positive upwards.  This means that index [0, 0, 0] is the
        southwest corner of the first layer.
        :param save_fn: Full path to save file to.
        :type save_fn: string or Path
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        # get resistivity model
        if self.res_model is None:
            self.res_model = np.zeros(
                (
                    self.nodes_north.size,
                    self.nodes_east.size,
                    self.nodes_z.size,
                )
            )
            self.res_model[:, :, :] = self.res_initial_value

        elif type(self.res_model) in [float, int]:
            self.res_initial_value = self.res_model
            self.res_model = np.zeros(
                (
                    self.nodes_north.size,
                    self.nodes_east.size,
                    self.nodes_z.size,
                )
            )
            self.res_model[:, :, :] = self.res_initial_value

        shift_east = (
            self.center_point.east
            - (
                self.nodes_east[0]
                - self.nodes_east[1] / 2
                - self.grid_center[1] / 2
            )
        ) / 1000.0
        shift_north = (
            self.center_point.north
            + (
                self.nodes_north[0]
                - self.nodes_north[1] / 2
                - self.grid_center[0] / 2
            )
        ) / 1000.0

        shift_elevation = self.center_point.elevation / 1000.0

        # --> write file
        with open(save_fn, "w") as ifid:
            ifid.write("\n")
            ifid.write(
                "{0:>5}{1:>5}{2:>5}{3:>5} {4}\n".format(
                    self.nodes_east.size,
                    self.nodes_north.size,
                    self.nodes_z.size,
                    0,
                    "VAL",
                )
            )

            # write S --> N node block
            for ii, nnode in enumerate(self.nodes_east):
                ifid.write(f"{abs(nnode):>12.3f}")

            ifid.write("\n")

            # write W --> E node block
            for jj, enode in enumerate(self.nodes_north):
                ifid.write(f"{abs(enode):>12.3f}")
            ifid.write("\n")

            # write top --> bottom node block
            for kk, zz in enumerate(self.nodes_z):
                ifid.write(f"{abs(zz):>12.3f}")
            ifid.write("\n")

            # write the resistivity in log e format
            write_res_model = self.res_model[::-1, :, :]

            # write out the layers from resmodel
            count = 1
            for zz in range(self.nodes_z.size):
                ifid.write(f"{count}\n")
                for nn in range(self.nodes_north.size):
                    for ee in range(self.nodes_east.size):
                        ifid.write(f"{write_res_model[nn, ee, zz]:>13.5E}")
                    ifid.write("\n")
                count += 1

            # write footer
            ifid.write("\n")
            ifid.write("WINGLINK\n")
            ifid.write("  Project      (site name)\n")
            ifid.write("           1           1    (i j block numbers)\n")
            ifid.write(
                f"   {shift_east:.3f}       {shift_north:.3f}       (real world coordinates)\n"
            )
            ifid.write("  0.0000000E+00    (rotation)\n")
            ifid.write(f"   {shift_elevation:.3f}       (top elevation)\n")
            ifid.write("\n")

        self._logger.info(f"Wrote file to: {save_fn}")

    def to_ubc(self, basename):
        """Write a UBC .msh and .mod file.
        :param basename:
        :param save_fn: DESCRIPTION.
        :type save_fn: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        # write mesh first
        lines = [
            f"{self.nodes_east.size} {self.nodes_north.size} {self.nodes_z.size}"
        ]
        lines.append(
            str(self.nodes_east.tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
        )
        lines.append(
            str(self.nodes_north.tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
        )
        lines.append(
            str(self.nodes_z.tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
        )

        with open(self.save_path.joinpath(basename + ".msh"), "w") as fid:
            fid.write("\n".join(lines))

    def convert_model_to_int(self, res_list=None):
        """Convert resistivity values to integers according to resistivity list.
        :param res_list: Resistivity values in Ohm-m, defaults to None.
        :type res_list: list of floats, optional
        :return: Array of integers corresponding to the res_list.
        :rtype: np.ndarray(dtype=int)
        """

        res_model_int = np.ones_like(self.res_model)
        if res_list is None:
            return res_model_int
        # make a dictionary of values to write to file.
        res_dict = dict(
            [(res, ii) for ii, res in enumerate(sorted(res_list), 1)]
        )

        for ii, res in enumerate(res_list):
            indexes = np.where(self.res_model == res)
            res_model_int[indexes] = res_dict[res]
            if ii == 0:
                indexes = np.where(self.res_model <= res)
                res_model_int[indexes] = res_dict[res]
            elif ii == len(res_list) - 1:
                indexes = np.where(self.res_model >= res)
                res_model_int[indexes] = res_dict[res]
            else:
                l_index = max([0, ii - 1])
                h_index = min([len(res_list) - 1, ii + 1])
                indexes = np.where(
                    (self.res_model > res_list[l_index])
                    & (self.res_model < res_list[h_index])
                )
                res_model_int[indexes] = res_dict[res]

        return res_model_int

    def to_ws3dinv_intial(self, initial_fn, res_list=None):
        """Write a WS3DINV inital model file."""

        # check to see what resistivity in input
        if res_list is None:
            nr = 0
        elif type(res_list) is not list and type(res_list) is not np.ndarray:
            res_list = [res_list]
            nr = len(res_list)
        else:
            nr = len(res_list)

        # --> write file
        lines = []
        lines.append(f"# {'Inital Model File made in MTpy'.upper()}\n")
        lines.append(
            "{0} {1} {2} {3}\n".format(
                self.nodes_north.shape[0],
                self.nodes_east.shape[0],
                self.nodes_z.shape[0],
                nr,
            )
        )

        # write S --> N node block
        for ii, nnode in enumerate(self.nodes_north):
            lines.append(f"{abs(nnode):>12.1f}")
            if ii != 0 and np.remainder(ii + 1, 5) == 0:
                lines.append("\n")
            elif ii == self.nodes_north.shape[0] - 1:
                lines.append("\n")

        # write W --> E node block
        for jj, enode in enumerate(self.nodes_east):
            lines.append(f"{abs(enode):>12.1f}")
            if jj != 0 and np.remainder(jj + 1, 5) == 0:
                lines.append("\n")
            elif jj == self.nodes_east.shape[0] - 1:
                lines.append("\n")

        # write top --> bottom node block
        for kk, zz in enumerate(self.nodes_z):
            lines.append(f"{abs(zz):>12.1f}")
            if kk != 0 and np.remainder(kk + 1, 5) == 0:
                lines.append("\n")
            elif kk == self.nodes_z.shape[0] - 1:
                lines.append("\n")

        # write the resistivity list
        if nr > 0:
            for ff in res_list:
                lines.append(f"{ff:.1f} ")
            lines.append("\n")
        else:
            pass

        if nr > 0:
            res_model_int = self.convert_model_to_int(res_list)
            # need to flip the array such that the 1st index written is the
            # northern most value
            write_res_model = res_model_int[::-1, :, :]
            # get similar layers
        else:
            write_res_model = self.res_model[::-1, :, :]
        l1 = 0
        layers = []
        for zz in range(self.nodes_z.shape[0] - 1):
            # if not (
            #     write_res_model[:, :, zz] == write_res_model[:, :, zz + 1]
            # ).all():
                layers.append((l1, zz))
                l1 = zz + 1
        # need to add on the bottom layers
        layers.append((l1, self.nodes_z.shape[0] - 1))

        # write out the layers from resmodel
        for ll in layers:
            if nr > 0:
                lines.append(f"{ll[0] + 1} {ll[1] + 1}\n")
            for nn in range(self.nodes_north.shape[0]):
                for ee in range(self.nodes_east.shape[0]):
                    if nr > 0:
                        lines.append(f"{write_res_model[nn, ee, ll[0]]:>3.0f}")
                    else:
                        lines.append(f"{write_res_model[nn, ee, ll[0]]:>8.1f}")
                lines.append("\n")

        with open(initial_fn, "w") as fid:
            fid.write("".join(lines))

        self._logger.info(f"Wrote WS3DINV intial model file to: {initial_fn}")

        return initial_fn

    def from_ws3dinv_initial(self, initial_fn):
        """Read an initial file and return the pertinent information including
        grid positions in coordinates relative to the center point (0,0) and
        starting model.

        Arguments::

                **initial_fn** : full path to initializing file.

        Outputs::

                **nodes_north** : np.array(nx)
                            array of nodes in S --> N direction

                **nodes_east** : np.array(ny)
                            array of nodes in the W --> E direction

                **nodes_z** : np.array(nz)
                            array of nodes in vertical direction positive downwards

                **res_model** : dictionary
                            dictionary of the starting model with keys as layers

                **res_list** : list
                            list of resistivity values in the model

                **title** : string
                             title string
        """

        with open(initial_fn, "r") as ifid:
            ilines = ifid.readlines()

        # get size of dimensions
        nsize = ilines[1].strip().split()
        n_north = int(nsize[0])
        n_east = int(nsize[1])
        n_z = int(nsize[2])
        # last integer describes resistivity format
        res_format = int(nsize[3])

        # initialize empy arrays to put things into
        self._nodes_north = np.zeros(n_north)
        self._nodes_east = np.zeros(n_east)
        self._nodes_z = np.zeros(n_z)
        res_model = np.zeros((n_north, n_east, n_z))

        # get the grid line locations
        line_index = 2  # line number in file
        count_n = 0  # number of north nodes found
        while count_n < n_north:
            iline = ilines[line_index].strip().split()
            for north_node in iline:
                self._nodes_north[count_n] = float(north_node)
                count_n += 1
            line_index += 1
        self.grid_north = np.insert(np.cumsum(self.nodes_north), 0, 0)
        self.grid_north = (
            self.grid_north - (self.grid_north[-1] - self.grid_north[0]) / 2
        )

        count_e = 0  # number of east nodes found
        while count_e < n_east:
            iline = ilines[line_index].strip().split()
            for east_node in iline:
                self._nodes_east[count_e] = float(east_node)
                count_e += 1
            line_index += 1
        self.grid_east = np.insert(np.cumsum(self.nodes_east), 0, 0)
        self.grid_east = (
            self.grid_east - (self.grid_east[-1] - self.grid_east[0]) / 2
        )

        count_z = 0  # number of vertical nodes
        while count_z < n_z:
            iline = ilines[line_index].strip().split()
            for z_node in iline:
                print(z_node)
                self._nodes_z[count_z] = float(z_node)
                count_z += 1
            line_index += 1

        self.grid_z = np.insert(np.cumsum(self._nodes_z), 0, 0)

        # get the resistivity values, if type > 1
        if res_format >= 1:
            res_list = [float(rr) for rr in ilines[line_index].strip().split()]
            line_index += 1
        else:
            res_list = []

        # read in model, according to format
        if res_format == 1:
            res_model[:, :, :] = res_list[0]
            return
        elif res_format > 1:
            while line_index < len(ilines):
                iline = ilines[line_index].strip().split()
                if len(iline) == 2:
                    l1 = int(iline[0]) - 1
                    l2 = int(iline[1])
                    if l1 == l2:
                        l2 += 1
                    line_index += 1
                    count_n = 0
                elif len(iline) == 0:
                    break
                else:
                    count_e = 0
                    while count_e < n_east:
                        # be sure the indes of res list starts at 0 not 1 as
                        # in ws3dinv
                        res_model[count_n, count_e, l1:l2] = res_list[
                            int(iline[count_e]) - 1
                        ]
                        count_e += 1
                    count_n += 1
                    line_index += 1
        elif res_format == 0:
            for count_z in range(n_z):
                for count_e in range(n_east):
                    for count_n in range(n_north):
                        iline = ilines[line_index].strip()
                        res_model[count_n, count_e, count_z] = float(iline)
                        line_index += 1

        # Need to be sure that the resistivity array matches
        # with the grids, such that the first index is the
        # furthest south, even though ws3dinv outputs as first
        # index as furthest north.
        self.res_model = res_model[::-1, :, :]

        return res_list

    def from_ws3dinv(self, model_fn):
        """Read WS3DINV iteration model file.
        :param model_fn: DESCRIPTION.
        :type model_fn: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        with open(model_fn, "r") as mfid:
            mlines = mfid.readlines()

        # get info at the beggining of file
        info = mlines[0].strip().split()
        iteration_number = int(info[2])
        rms = float(info[5])
        try:
            lagrange = float(info[8])
        except IndexError:
            self._logger.warning("Did not get Lagrange Multiplier")
            lagrange = 1.0

        # get lengths of things
        n_north, n_east, n_z, n_res = np.array(
            mlines[1].strip().split(), dtype=int
        )

        # make empty arrays to put stuff into
        self._nodes_north = np.zeros(n_north)
        self._nodes_east = np.zeros(n_east)
        self.grid_z = np.zeros(n_z + 1)
        self.res_model = np.zeros((n_north, n_east, n_z))

        # get the grid line locations
        line_index = 2  # line number in file
        count_n = 0  # number of north nodes found
        while count_n < n_north:
            mline = mlines[line_index].strip().split()
            for north_node in mline:
                self._nodes_north[count_n] = float(north_node)
                count_n += 1
            line_index += 1
        self.grid_north = np.insert(np.cumsum(self.nodes_north), 0, 0)
        self.grid_north = (
            self.grid_north - (self.grid_north[-1] - self.grid_north[0]) / 2
        )

        count_e = 0  # number of east nodes found
        while count_e < n_east:
            mline = mlines[line_index].strip().split()
            for east_node in mline:
                self._nodes_east[count_e] = float(east_node)
                count_e += 1
            line_index += 1
        self.grid_east = np.insert(np.cumsum(self.nodes_east), 0, 0)
        self.grid_east = (
            self.grid_east - (self.grid_east[-1] - self.grid_east[0]) / 2
        )

        count_z = 0  # number of vertical nodes
        zdep = 0
        while count_z < n_z:
            mline = mlines[line_index].strip().split()
            for z_node in mline:
                self.grid_z[count_z] = zdep
                count_z += 1
                zdep += float(z_node)
            line_index += 1
        self.grid_z[count_z] = zdep

        # --> get resistivity values
        # need to read in the north backwards so that the first index is
        # southern most point
        for kk in range(n_z):
            for jj in range(n_east):
                for ii in range(n_north):
                    self.res_model[(n_north - 1) - ii, jj, kk] = float(
                        mlines[line_index].strip()
                    )
                    line_index += 1

        return {
            "rms": rms,
            "lagrange_mulitplier": lagrange,
            "info": info,
            "iteration": iteration_number,
        }

    def estimate_skin_depth(self, apparent_resistivity, period, scale="km"):
        """Estimate skin depth from apparent resistivity and period.
        :param apparent_resistivity: DESCRIPTION.
        :type apparent_resistivity: TYPE
        :param period: DESCRIPTION.
        :type period: TYPE
        :param scale: DESCRIPTION, defaults to "km".
        :type scale: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if scale in ["km", "kilometers"]:
            dscale = 1000
        elif scale in ["m", "meters"]:
            dscale = 1
        elif scale in ["ft", "feet"]:
            dscale = 3.2808399
        else:
            raise ValueError(f"Could not understand scale {scale}.")
        return 503 * np.sqrt(apparent_resistivity * period) / dscale
