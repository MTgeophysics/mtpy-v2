# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:15:37 2014

@author: jrpeacock
"""

# =============================================================================
# Imports
# =============================================================================
try:
    from osgeo import ogr, gdal, osr
except ImportError:
    raise ImportError(
        "Did not find GDAL, be sure it is installed correctly and "
        "all the paths are correct."
    )
import numpy as np
from mtpy import MTLocation

ogr.UseExceptions()
# =============================================================================

# class ModEM_to_Raster(object):
#     """
#     create a raster image of a model slice from a ModEM model

#     :Example: ::
#         >>> import mtpy.utils.array2raster as a2r
#         >>> mfn = r"/home/ModEM/Inv1/Modular_NLCG_110.rho"
#         >>> m_obj = a2r.ModEM_to_Raster()
#         >>> m_obj.model_fn = mfn
#         >>> m_obj.lower_left_corner = (-119.11, 37.80)
#         >>> m_obj.write_raster_files(save_path=r"/home/ModEM/Inv1/GIS_depth_slices")


#     """

#     def __init__(self, **kwargs):
#         self.model_fn = kwargs.pop("model_fn", None)
#         self.save_path = kwargs.pop("save_path", os.getcwd())
#         self.projection = kwargs.pop("projection", "WGS84")
#         self.lower_left_corner = kwargs.pop("lower_left_corner", None)
#         self.grid_center = kwargs.pop("grid_center", None)

#         self.pad_east = None
#         self.pad_north = None
#         self.res_array = None
#         self.cell_size_east = None
#         self.cell_size_north = None
#         self.rotation_angle = 0
#         self.model_obj = modem.Model()

#     def _get_model(self):
#         """
#         get model to put into array
#         """

#         self.model_obj = mtpy.modeling.modem.Model()
#         self.model_obj.model_fn = self.model_fn
#         self.model_obj.read_model_file()

#         self.cell_size_east = np.median(self.model_obj.nodes_east)
#         self.cell_size_north = np.median(self.model_obj.nodes_north)

#         self.pad_east = np.where(
#             self.model_obj.nodes_east[0:10] > self.cell_size_east * 1.1
#         )[0][-1]
#         self.pad_north = np.where(
#             self.model_obj.nodes_north[0:10] > self.cell_size_north * 1.1
#         )[0][-1]
#         self.grid_z = self.model_obj.grid_z.copy()
#         self.res_array = self.model_obj.res_model[
#             self.pad_north : -self.pad_north, self.pad_east : -self.pad_east, :
#         ]

#     def get_model_lower_left_coord(
#         self, model_center=None, pad_east=0, pad_north=0
#     ):
#         """
#         Find the models lower left hand corner in (lon, lat) decimal degrees
#         """

#         self.pad_east = pad_east
#         self.pad_north = pad_north

#         if model_center:
#             (
#                 center_east,
#                 center_north,
#                 center_zone,
#             ) = gis_tools.project_point_ll2utm(model_center[0], model_center[1])

#             print(center_east, center_north, center_zone)
#             lower_left_east = (
#                 center_east
#                 + self.model_obj.grid_center[1]
#                 + self.model_obj.nodes_east[0:pad_east].sum()
#                 - self.model_obj.nodes_east[pad_east] / 2
#             )
#             lower_left_north = (
#                 center_north
#                 + self.model_obj.grid_center[1]
#                 + self.model_obj.nodes_north[0:pad_north].sum()
#                 + self.model_obj.nodes_north[pad_north] / 2
#             )

#             ll_lat, ll_lon = gis_tools.project_point_utm2ll(
#                 lower_left_east, lower_left_north, str(center_zone)
#             )

#             print(
#                 "Lower Left Coordinates should be ({0:.5f}, {1:.5f})".format(
#                     ll_lon, ll_lat
#                 )
#             )
#             return (ll_lon, ll_lat)
#         else:
#             raise IOError("Need to input model center (lon, lat)")

#     def interpolate_grid(self, pad_east=None, pad_north=None, cell_size=None):
#         """
#         interpolate the irregular model grid onto a regular grid.

#         """

#         self.grid_z = self.model_obj.grid_z.copy()

#         if cell_size is not None:
#             self.cell_size_east = cell_size
#             self.cell_size_north = cell_size
#         else:
#             self.cell_size_east = np.median(self.model_obj.nodes_east)
#             self.cell_size_north = np.median(self.model_obj.nodes_north)

#         if pad_east is not None:
#             self.pad_east = pad_east

#         if self.pad_east is None:
#             self.pad_east = np.where(
#                 self.model_obj.nodes_east[0:25] > self.cell_size_east * 1.1
#             )[0][-1]
#         if pad_north is not None:
#             self.pad_north = pad_north
#         if self.pad_north is None:
#             self.pad_north = np.where(
#                 self.model_obj.nodes_north[0:25] > self.cell_size_north * 1.1
#             )[0][-1]

#         print("Pad north = {0}".format(self.pad_north))
#         print("Pad east  = {0}".format(self.pad_east))

#         new_east = np.arange(
#             self.model_obj.grid_east[self.pad_east],
#             self.model_obj.grid_east[-self.pad_east - 2],
#             self.cell_size_east,
#         )
#         new_north = np.arange(
#             self.model_obj.grid_north[self.pad_north],
#             self.model_obj.grid_north[-self.pad_north - 2],
#             self.cell_size_north,
#         )

#         # needs to be -1 because the grid is n+1 as it is the edges of the
#         # the nodes.  Might need to change this in the future
#         model_n, model_e = np.broadcast_arrays(
#             self.model_obj.grid_north[:-1, None],
#             self.model_obj.grid_east[None, :-1],
#         )

#         new_res_arr = np.zeros(
#             (new_north.size, new_east.size, self.model_obj.nodes_z.size)
#         )

#         for z_index in range(self.model_obj.grid_z.shape[0] - 1):
#             res = self.model_obj.res_model[:, :, z_index]
#             new_res_arr[:, :, z_index] = interpolate.griddata(
#                 (model_n.ravel(), model_e.ravel()),
#                 res.ravel(),
#                 (new_north[:, None], new_east[None, :]),
#             )

#         self.res_array = new_res_arr

#     def write_raster_files(
#         self,
#         save_path=None,
#         pad_east=None,
#         pad_north=None,
#         cell_size=None,
#         rotation_angle=None,
#     ):
#         """
#         write a raster file for each layer

#         """
#         if rotation_angle is not None:
#             self.rotation_angle = float(rotation_angle)

#         if self.lower_left_corner is None:
#             raise ValueError("Need to input an lower_left_corner as (lon, lat)")
#         if save_path is not None:
#             self.save_path = save_path

#         if not os.path.exists(self.save_path):
#             os.mkdir(self.save_path)

#         self.interpolate_grid(
#             pad_east=pad_east, pad_north=pad_north, cell_size=cell_size
#         )

#         for ii in range(self.res_array.shape[2]):
#             d = self.grid_z[ii]
#             raster_fn = os.path.join(
#                 self.save_path,
#                 "Depth_{0:.2f}_{1}.tif".format(d, self.projection),
#             )
#             array2raster(
#                 raster_fn,
#                 self.lower_left_corner,
#                 self.cell_size_east,
#                 self.cell_size_north,
#                 np.log10(self.res_array[:, :, ii]),
#                 projection=self.projection,
#                 rotation_angle=self.rotation_angle,
#             )
#             print(
#                 os.path.join(
#                     self.save_path,
#                     "Depth_{0:.2f}_{1}.tif".format(d, self.projection),
#                 )
#             )


# # ==============================================================================
# #  WS3dInv to raster
# # ==============================================================================
# class WS3D_to_Raster(object):
#     """
#     create a raster image of a model slice from a ModEM model

#     :Example: ::
#         >>> import mtpy.utils.array2raster as a2r
#         >>> mfn = r"/home/ModEM/Inv1/Modular_NLCG_110.rho"
#         >>> m_obj = a2r.WS_to_Raster()
#         >>> m_obj.model_fn = mfn
#         >>> m_obj.lower_left_corner = (-119.11, 37.80)
#         >>> m_obj.write_raster_files(save_path=r"/home/WS3DINV/Inv1/GIS_depth_slices")


#     """

#     def __init__(self, **kwargs):
#         self.model_fn = kwargs.pop("model_fn", None)
#         self.save_path = kwargs.pop("save_path", os.getcwd())
#         self.projection = kwargs.pop("projection", "WGS84")
#         self.lower_left_corner = kwargs.pop("lower_left_corner", None)

#         self.pad_east = None
#         self.pad_north = None
#         self.res_array = None
#         self.cell_size_east = None
#         self.cell_size_north = None
#         self.rotation_angle = 0

#     def _get_model(self):
#         """
#         get model to put into array
#         """

#         model_obj = ws.WSModel()
#         self.model_obj.model_fn = self.model_fn
#         self.model_obj.read_model_file()

#         self.cell_size_east = np.median(self.model_obj.nodes_east)
#         self.cell_size_north = np.median(self.model_obj.nodes_north)

#         self.pad_east = np.where(
#             self.model_obj.nodes_east[0:10] > self.cell_size_east * 1.1
#         )[0][-1]
#         self.pad_north = np.where(
#             self.model_obj.nodes_north[0:10] > self.cell_size_north * 1.1
#         )[0][-1]
#         self.grid_z = self.model_obj.grid_z.copy()
#         self.res_array = self.model_obj.res_model[
#             self.pad_north : -self.pad_north, self.pad_east : -self.pad_east, :
#         ]

#     def interpolate_grid(self, pad_east=None, pad_north=None, cell_size=None):
#         """
#         interpolate the irregular model grid onto a regular grid.

#         """

#         model_obj = ws.WSModel()
#         model_obj.model_fn = self.model_fn
#         model_obj.read_model_file()

#         self.grid_z = model_obj.grid_z.copy()

#         if cell_size is not None:
#             self.cell_size_east = cell_size
#             self.cell_size_north = cell_size
#         else:
#             self.cell_size_east = np.median(model_obj.nodes_east)
#             self.cell_size_north = np.median(model_obj.nodes_north)

#         if pad_east is not None:
#             self.pad_east = pad_east

#         if self.pad_east is None:
#             self.pad_east = np.where(
#                 model_obj.nodes_east[0:10] > self.cell_size_east * 1.1
#             )[0][-1]
#         if pad_north is not None:
#             self.pad_north = pad_north
#         if self.pad_north is None:
#             self.pad_north = np.where(
#                 model_obj.nodes_north[0:10] > self.cell_size_north * 1.1
#             )[0][-1]

#         new_east = np.arange(
#             model_obj.grid_east[self.pad_east],
#             model_obj.grid_east[-self.pad_east - 1],
#             self.cell_size_east,
#         )
#         new_north = np.arange(
#             model_obj.grid_north[self.pad_north],
#             model_obj.grid_north[-self.pad_north - 1],
#             self.cell_size_north,
#         )

#         model_n, model_e = np.broadcast_arrays(
#             model_obj.grid_north[:, None], model_obj.grid_east[None, :]
#         )

#         new_res_arr = np.zeros(
#             (new_north.shape[0], new_east.shape[0], model_obj.grid_z.shape[0])
#         )

#         for z_index in range(model_obj.grid_z.shape[0]):
#             res = model_obj.res_model[:, :, z_index]
#             new_res_arr[:, :, z_index] = interpolate.griddata(
#                 (model_n.ravel(), model_e.ravel()),
#                 res.ravel(),
#                 (new_north[:, None], new_east[None, :]),
#             )

#         self.res_array = new_res_arr

#     def write_raster_files(
#         self,
#         save_path=None,
#         pad_east=None,
#         pad_north=None,
#         cell_size=None,
#         rotation_angle=None,
#     ):
#         """
#         write a raster file for each layer

#         """
#         if rotation_angle is not None:
#             self.rotation_angle = rotation_angle

#         if self.lower_left_corner is None:
#             raise ValueError("Need to input an lower_left_corner as (lon, lat)")
#         if save_path is not None:
#             self.save_path = save_path

#         if not os.path.exists(self.save_path):
#             os.mkdir(self.save_path)

#         self.interpolate_grid(
#             pad_east=pad_east, pad_north=pad_north, cell_size=cell_size
#         )

#         for ii in range(self.res_array.shape[2]):
#             d = self.grid_z[ii]
#             raster_fn = os.path.join(
#                 self.save_path,
#                 "Depth_{0:.2f}_{1}.tif".format(d, self.projection),
#             )
#             array2raster(
#                 raster_fn,
#                 self.lower_left_corner,
#                 self.cell_size_east,
#                 self.cell_size_north,
#                 np.log10(self.res_array[:, :, ii]),
#                 projection=self.projection,
#                 rotation_angle=self.rotation_angle,
#             )


# ==============================================================================
# create a raster from an array
# ==============================================================================


def array2raster(
    raster_fn,
    utm_lower_left_mt_location,
    cell_width,
    cell_height,
    res_array,
    utm_epsg,
    rotation_angle=0.0,
):
    """Converts an array into a raster file that can be read into a GIS program.

    utm_lower_left_mt_location should be a MTLocation object with a UTM
    projection and represents the lower left hand corner of the grid.

    Arguments::
            **raster_fn** : string
                            full path to save raster file to

            **origin** : (lon, lat)
                         longitude and latitude of southwest corner of the array

            **cell_width** : float (in meters)
                             size of model cells in east-west direction

            **cell_height** : float (in meters)
                             size of model cells in north-south direction

            **res_array** : np.ndarray(east, north)
                            resistivity array in linear scale.

            **projection** : string
                            name of the projection datum

            **rotation_angle** : float
                                 angle in degrees to rotate grid.
                                 Assuming N = 0 and E = 90, positive clockwise

    Output::
            * creates a geotiff file projected into projection in UTM.  The
              values are in log scale for coloring purposes.
    """
    # convert rotation angle to radians
    r_theta = np.deg2rad(rotation_angle)

    res_array = np.flipud(res_array[::-1])

    ncols = res_array.shape[1]
    nrows = res_array.shape[0]

    if not isinstance(utm_lower_left_mt_location, MTLocation):
        raise TypeError(
            "utm_lower_left_mt_location must be type mtpy.MTLocation, not "
            f"{type(utm_lower_left_mt_location)}."
        )
    ll_origin = utm_lower_left_mt_location.copy()
    if ll_origin.utm_crs is None:
        raise ValueError("Must set UTM CRS")

    # set drive to make a geo tiff
    driver = gdal.GetDriverByName("GTiff")

    # make a raster with the shape of the array to be written
    out_raster = driver.Create(raster_fn, ncols, nrows, 1, gdal.GDT_Float32)

    out_raster.SetGeoTransform(
        (
            ll_origin.east,
            np.cos(r_theta) * cell_width,
            -np.sin(r_theta) * cell_width,
            ll_origin.north,
            np.sin(r_theta) * cell_height,
            np.cos(r_theta) * cell_height,
        )
    )

    # create a band for the raster data to be put in
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(res_array)

    # geo reference the raster
    utm_cs = osr.SpatialReference(wkt=ll_origin.utm_crs.to_wkt())

    out_raster.SetProjection(utm_cs.ExportToWkt())

    # be sure to flush the data
    outband.FlushCache()
