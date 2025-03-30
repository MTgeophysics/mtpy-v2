# ==============================================================================
import numpy as np
import scipy as sp
from scipy.stats import mode
import os
import os.path as op
import time
import matplotlib.colorbar as mcb
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm

from mtpy.modeling.occam2d.model import Occam2DModel as Model
from mtpy.modeling.occam2d.data import Occam2DData as Data

# ==============================================================================



class PlotModel(Model):
    """
    plot the 2D model found by Occam2D.  The model is displayed as a meshgrid
    instead of model bricks.  This speeds things up considerably.  
    
    Inherets the Model class to take advantage of the attributes and methods
    already coded.
    
    Arguments:
    -----------
        **iter_fn** : string
                      full path to iteration file.  From here all the 
                      necessary files can be found assuming they are in the 
                      same directory.  If they are not then need to input
                      manually.
    
    
    ======================= ===============================================
    keywords                description
    ======================= ===============================================
    block_font_size         font size of block number is blocknum == 'on'
    blocknum                [ 'on' | 'off' ] to plot regulariztion block 
                            numbers.
    cb_pad                  padding between axes edge and color bar 
    cb_shrink               percentage to shrink the color bar
    climits                 limits of the color scale for resistivity
                            in log scale (min, max)
    cmap                    name of color map for resistivity values
    femesh                  plot the finite element mesh
    femesh_triangles        plot the finite element mesh with each block
                            divided into four triangles
    fig_aspect              aspect ratio between width and height of 
                            resistivity image. 1 for equal axes
    fig_dpi                 resolution of figure in dots-per-inch
    fig_num                 number of figure instance
    fig_size                size of figure in inches (width, height)
    font_size               size of axes tick labels, axes labels is +2
    grid                    [ 'both' | 'major' |'minor' | None ] string 
                            to tell the program to make a grid on the 
                            specified axes.
    meshnum                 [ 'on' | 'off' ] 'on' will plot finite element
                            mesh numbers
    meshnum_font_size       font size of mesh numbers if meshnum == 'on'
    ms                      size of station marker 
    plot_yn                 [ 'y' | 'n']
                            'y' --> to plot on instantiation
                            'n' --> to not plot on instantiation
    regmesh                 [ 'on' | 'off' ] plot the regularization mesh
                            plots as blue lines
    station_color           color of station marker
    station_font_color      color station label
    station_font_pad        padding between station label and marker
    station_font_rotation   angle of station label in degrees 0 is 
                            horizontal
    station_font_size       font size of station label
    station_font_weight     font weight of station label
    station_id              index to take station label from station name
    station_marker          station marker.  if inputing a LaTex marker
                            be sure to input as r"LaTexMarker" otherwise
                            might not plot properly
    subplot_bottom          subplot spacing from bottom  
    subplot_left            subplot spacing from left  
    subplot_right           subplot spacing from right
    subplot_top             subplot spacing from top
    title                   title of plot.  If None then the name of the
                            iteration file and containing folder will be
                            the title with RMS and Roughness.
    xlimits                 limits of plot in x-direction in (km) 
    xminorticks             increment of minor ticks in x direction
    xpad                    padding in x-direction in km
    ylimits                 depth limits of plot positive down (km)
    yminorticks             increment of minor ticks in y-direction
    ypad                    padding in negative y-direction (km)
    yscale                  [ 'km' | 'm' ] scale of plot, if 'm' everything
                            will be scaled accordingly.
    ======================= ===============================================
    
    =================== =======================================================
    Methods             Description
    =================== =======================================================
    plot                plots resistivity model.  
    redraw_plot         call redraw_plot to redraw the figures, 
                        if one of the attributes has been changed
    save_figure         saves the matplotlib.figure instance to desired 
                        location and format
    =================== ======================================================
    
    :Example: 
    ---------------
        >>> import mtpy.modeling.occam2d as occam2d
        >>> model_plot = occam2d.PlotModel(r"/home/occam/Inv1/mt_01.iter")
        >>> # change the color limits
        >>> model_plot.climits = (1, 4)
        >>> model_plot.redraw_plot()
        >>> #change len of station name
        >>> model_plot.station_id = [2, 5]
        >>> model_plot.redraw_plot()
        
    
    """

    def __init__(self, iter_fn=None, data_fn=None, **kwargs):
        Model.__init__(self, iter_fn, **kwargs)

        self.yscale = kwargs.pop('yscale', 'km')

        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.fig_aspect = kwargs.pop('fig_aspect', 1)
        self.title = kwargs.pop('title', 'on')

        self.xpad = kwargs.pop('xpad', 1.0)
        self.ypad = kwargs.pop('ypad', 1.0)

        self.ms = kwargs.pop('ms', 10)

        self.station_locations = None
        self.station_list = None
        self.station_id = kwargs.pop('station_id', None)
        self.station_font_size = kwargs.pop('station_font_size', 8)
        self.station_font_pad = kwargs.pop('station_font_pad', 1.0)
        self.station_font_weight = kwargs.pop('station_font_weight', 'bold')
        self.station_font_rotation = kwargs.pop('station_font_rotation', 60)
        self.station_font_color = kwargs.pop('station_font_color', 'k')
        self.station_marker = kwargs.pop('station_marker',
                                         r"$\blacktriangledown$")
        self.station_color = kwargs.pop('station_color', 'k')

        self.ylimits = kwargs.pop('ylimits', None)
        self.xlimits = kwargs.pop('xlimits', None)

        self.xminorticks = kwargs.pop('xminorticks', 5)
        self.yminorticks = kwargs.pop('yminorticks', 1)

        self.climits = kwargs.pop('climits', (0, 4))
        self.cmap = kwargs.pop('cmap', 'jet_r')
        if type(self.cmap) == str:
            self.cmap = cm.get_cmap(self.cmap)
        self.font_size = kwargs.pop('font_size', 8)

        self.femesh = kwargs.pop('femesh', 'off')
        self.femesh_triangles = kwargs.pop('femesh_triangles', 'off')
        self.femesh_lw = kwargs.pop('femesh_lw', .4)
        self.femesh_color = kwargs.pop('femesh_color', 'k')
        self.meshnum = kwargs.pop('meshnum', 'off')
        self.meshnum_font_size = kwargs.pop('meshnum_font_size', 3)

        self.regmesh = kwargs.pop('regmesh', 'off')
        self.regmesh_lw = kwargs.pop('regmesh_lw', .4)
        self.regmesh_color = kwargs.pop('regmesh_color', 'b')
        self.blocknum = kwargs.pop('blocknum', 'off')
        self.block_font_size = kwargs.pop('block_font_size', 3)
        self.grid = kwargs.pop('grid', None)

        self.cb_shrink = kwargs.pop('cb_shrink', .8)
        self.cb_pad = kwargs.pop('cb_pad', .01)

        self.subplot_right = .99
        self.subplot_left = .085
        self.subplot_top = .92
        self.subplot_bottom = .1

        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()

    def plot(self):
        """
        plotModel will plot the model output by occam2d in the iteration file.
        
        
        :Example: ::
            
            >>> import mtpy.modeling.occam2d as occam2d
            >>> itfn = r"/home/Occam2D/Line1/Inv1/Test_15.iter"
            >>> model_plot = occam2d.PlotModel(itfn)
            >>> model_plot.ms = 20
            >>> model_plot.ylimits = (0,.350)
            >>> model_plot.yscale = 'm'
            >>> model_plot.spad = .10
            >>> model_plot.ypad = .125
            >>> model_plot.xpad = .025
            >>> model_plot.climits = (0,2.5)
            >>> model_plot.aspect = 'equal'
            >>> model_plot.redraw_plot()
            
        """
        # --> read in iteration file and build the model
        self.read_iter_file()
        self.build_model()

        # --> get station locations and names from data file
        d_object = Data()
        d_object.read_data_file(self.data_fn)
        setattr(self, 'station_locations', d_object.station_locations.copy())
        setattr(self, 'station_list', d_object.station_list.copy())

        # set the scale of the plot
        if self.yscale == 'km':
            df = 1000.
            pf = 1.0
        elif self.yscale == 'm':
            df = 1.
            pf = 1000.
        else:
            df = 1000.
            pf = 1.0

        # set some figure properties to use the maiximum space
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.subplot.left'] = self.subplot_left
        plt.rcParams['figure.subplot.right'] = self.subplot_right
        plt.rcParams['figure.subplot.bottom'] = self.subplot_bottom
        plt.rcParams['figure.subplot.top'] = self.subplot_top

        # station font dictionary
        fdict = {'size': self.station_font_size,
                 'weight': self.station_font_weight,
                 'rotation': self.station_font_rotation,
                 'color': self.station_font_color}

        # plot the model as a mesh
        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()

        # add a subplot to the figure with the specified aspect ratio
        ax = self.fig.add_subplot(1, 1, 1, aspect=self.fig_aspect)

        # plot the model as a pcolormesh so the extents are constrained to
        # the model coordinates
        ax.pcolormesh(self.mesh_x / df,
                      self.mesh_z / df,
                      self.res_model,
                      cmap=self.cmap,
                      vmin=self.climits[0],
                      vmax=self.climits[1])

        # make a colorbar for the resistivity
        cbx = mcb.make_axes(ax, shrink=self.cb_shrink, pad=self.cb_pad)
        cb = mcb.ColorbarBase(cbx[0],
                              cmap=self.cmap,
                              norm=Normalize(vmin=self.climits[0],
                                             vmax=self.climits[1]))

        cb.set_label('Resistivity ($\Omega \cdot$m)',
                     fontdict={'size': self.font_size + 1, 'weight': 'bold'})
        cb.set_ticks(np.arange(int(self.climits[0]), int(self.climits[1]) + 1))
        cb.set_ticklabels(['10$^{0}$'.format('{' + str(nn) + '}') for nn in
                           np.arange(int(self.climits[0]),
                                     int(self.climits[1]) + 1)])

        # set the offsets of the stations and plot the stations
        # need to figure out a way to set the marker at the surface in all
        # views.
        for offset, name in zip(self.station_locations, self.station_list):
            # plot the station marker
            # plots a V for the station cause when you use scatter the spacing
            # is variable if you change the limits of the y axis, this way it
            # always plots at the surface.
            ax.text(offset / df,
                    self.plot_z.min(),
                    self.station_marker,
                    horizontalalignment='center',
                    verticalalignment='baseline',
                    fontdict={'size': self.ms, 'color': self.station_color})

            # put station id onto station marker
            # if there is a station id index
            if self.station_id != None:
                ax.text(offset / df,
                        -self.station_font_pad * pf,
                        name[self.station_id[0]:self.station_id[1]],
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontdict=fdict)
            # otherwise put on the full station name found form data file
            else:
                ax.text(offset / df,
                        -self.station_font_pad * pf,
                        name,
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontdict=fdict)

        # set the initial limits of the plot to be square about the profile line
        if self.ylimits == None:
            ax.set_ylim(abs(self.station_locations.max() -
                            self.station_locations.min()) / df,
                        -self.ypad * pf)
        else:
            ax.set_ylim(self.ylimits[1] * pf,
                        (self.ylimits[0] - self.ypad) * pf)
        if self.xlimits == None:
            ax.set_xlim(self.station_locations.min() / df - (self.xpad * pf),
                        self.station_locations.max() / df + (self.xpad * pf))
        else:
            ax.set_xlim(self.xlimits[0] * pf, self.xlimits[1] * pf)

        # set the axis properties
        ax.xaxis.set_minor_locator(MultipleLocator(self.xminorticks * pf))
        ax.yaxis.set_minor_locator(MultipleLocator(self.yminorticks * pf))

        # set axes labels
        ax.set_xlabel('Horizontal Distance ({0})'.format(self.yscale),
                      fontdict={'size': self.font_size + 2, 'weight': 'bold'})
        ax.set_ylabel('Depth ({0})'.format(self.yscale),
                      fontdict={'size': self.font_size + 2, 'weight': 'bold'})

        # put a grid on if one is desired
        if self.grid is not None:
            ax.grid(alpha=.3, which=self.grid, lw=.35)

        # set title as rms and roughness
        if type(self.title) is str:
            if self.title == 'on':
                titlestr = os.path.join(os.path.basename(
                    os.path.dirname(self.iter_fn)),
                    os.path.basename(self.iter_fn))
                ax.set_title('{0}: RMS={1:.2f}, Roughness={2:.0f}'.format(
                    titlestr, self.misfit_value, self.roughness_value),
                    fontdict={'size': self.font_size + 1,
                              'weight': 'bold'})
            else:
                ax.set_title('{0}; RMS={1:.2f}, Roughness={2:.0f}'.format(
                    self.title, self.misfit_value,
                    self.roughness_value),
                    fontdict={'size': self.font_size + 1,
                              'weight': 'bold'})
        else:
            print('RMS {0:.2f}, Roughness={1:.0f}'.format(self.misfit_value,
                                                          self.roughness_value))

        # plot forward model mesh
        # making an extended list seperated by None's speeds up the plotting
        # by as much as 99 percent, handy
        if self.femesh == 'on':
            row_line_xlist = []
            row_line_ylist = []
            for xx in self.plot_x / df:
                row_line_xlist.extend([xx, xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.plot_zy[0] / df])
                row_line_ylist.append(None)

            # plot column lines (variables are a little bit of a misnomer)
            ax.plot(row_line_xlist,
                    row_line_ylist,
                    color='k',
                    lw=.5)

            col_line_xlist = []
            col_line_ylist = []
            for yy in self.plot_z / df:
                col_line_xlist.extend([self.plot_x[0] / df,
                                       self.plot_x[-1] / df])
                col_line_xlist.append(None)
                col_line_ylist.extend([yy, yy])
                col_line_ylist.append(None)

            # plot row lines (variables are a little bit of a misnomer)
            ax.plot(col_line_xlist,
                    col_line_ylist,
                    color='k',
                    lw=.5)

        if self.femesh_triangles == 'on':
            row_line_xlist = []
            row_line_ylist = []
            for xx in self.plot_x / df:
                row_line_xlist.extend([xx, xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.plot_z[0] / df])
                row_line_ylist.append(None)

            # plot columns
            ax.plot(row_line_xlist,
                    row_line_ylist,
                    color='k',
                    lw=.5)

            col_line_xlist = []
            col_line_ylist = []
            for yy in self.plot_z / df:
                col_line_xlist.extend([self.plot_x[0] / df,
                                       self.plot_x[-1] / df])
                col_line_xlist.append(None)
                col_line_ylist.extend([yy, yy])
                col_line_ylist.append(None)

            # plot rows
            ax.plot(col_line_xlist,
                    col_line_ylist,
                    color='k',
                    lw=.5)

            diag_line_xlist = []
            diag_line_ylist = []
            for xi, xx in enumerate(self.plot_x[:-1] / df):
                for yi, yy in enumerate(self.plot_z[:-1] / df):
                    diag_line_xlist.extend([xx, self.plot_x[xi + 1] / df])
                    diag_line_xlist.append(None)
                    diag_line_xlist.extend([xx, self.plot_x[xi + 1] / df])
                    diag_line_xlist.append(None)

                    diag_line_ylist.extend([yy, self.plot_z[yi + 1] / df])
                    diag_line_ylist.append(None)
                    diag_line_ylist.extend([self.plot_z[yi + 1] / df, yy])
                    diag_line_ylist.append(None)

            # plot diagonal lines.
            ax.plot(diag_line_xlist,
                    diag_line_ylist,
                    color='k',
                    lw=.5)

        # plot the regularization mesh
        if self.regmesh == 'on':
            line_list = []
            for ii in range(len(self.model_rows)):
                # get the number of layers to combine
                # this index will be the first index in the vertical direction
                ny1 = self.model_rows[:ii, 0].sum()

                # the second index  in the vertical direction
                ny2 = ny1 + self.model_rows[ii][0]

                # make the list of amalgamated columns an array for ease
                lc = np.array(self.model_columns[ii])
                yline = ax.plot([self.plot_x[0] / df, self.plot_x[-1] / df],
                                [self.plot_z[-ny1] / df,
                                 self.plot_z[-ny1] / df],
                                color='b',
                                lw=.5)

                line_list.append(yline)

                # loop over the number of amalgamated blocks
                for jj in range(len(self.model_columns[ii])):
                    # get first in index in the horizontal direction
                    nx1 = lc[:jj].sum()

                    # get second index in horizontal direction
                    nx2 = nx1 + lc[jj]
                    try:
                        if ny1 == 0:
                            ny1 = 1
                        xline = ax.plot([self.plot_x[nx1] / df,
                                         self.plot_x[nx1] / df],
                                        [self.plot_z[-ny1] / df,
                                         self.plot_z[-ny2] / df],
                                        color='b',
                                        lw=.5)
                        line_list.append(xline)
                    except IndexError:
                        pass

        ##plot the mesh block numbers
        if self.meshnum == 'on':
            kk = 1
            for yy in self.plot_z[::-1] / df:
                for xx in self.plot_x / df:
                    ax.text(xx, yy, '{0}'.format(kk),
                            fontdict={'size': self.meshnum_font_size})
                    kk += 1

        ##plot regularization block numbers
        if self.blocknum == 'on':
            kk = 1
            for ii in range(len(self.model_rows)):
                # get the number of layers to combine
                # this index will be the first index in the vertical direction
                ny1 = self.model_rows[:ii, 0].sum()

                # the second index  in the vertical direction
                ny2 = ny1 + self.model_rows[ii][0]
                # make the list of amalgamated columns an array for ease
                lc = np.array(self.model_cols[ii])
                # loop over the number of amalgamated blocks
                for jj in range(len(self.model_cols[ii])):
                    # get first in index in the horizontal direction
                    nx1 = lc[:jj].sum()
                    # get second index in horizontal direction
                    nx2 = nx1 + lc[jj]
                    try:
                        if ny1 == 0:
                            ny1 = 1
                        # get center points of the blocks
                        yy = self.plot_z[-ny1] - (self.plot_z[-ny1] -
                                                  self.plot_z[-ny2]) / 2
                        xx = self.plot_x[nx1] - \
                             (self.plot_x[nx1] - self.plot_x[nx2]) / 2
                        # put the number
                        ax.text(xx / df, yy / df, '{0}'.format(kk),
                                fontdict={'size': self.block_font_size},
                                horizontalalignment='center',
                                verticalalignment='center')
                        kk += 1
                    except IndexError:
                        pass

        plt.show()

        # make attributes that can be manipulated
        self.ax = ax
        self.cbax = cb
