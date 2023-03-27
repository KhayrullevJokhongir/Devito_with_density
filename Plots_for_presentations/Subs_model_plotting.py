import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import concurrent.futures as cf
import pyvista as pv
import segyio

from scipy import interpolate



#######################################################################################
# Used to read segy files

def read_segy_file(file):
    """ Open a segy file, read data inside of the file and get file shape"""
    # Note:
    # file = file name (including its directory if it is not at the same place with this code) in quotes

    segy_file = file

    read_segy_file = segyio.open(segy_file, mode="r+", iline=81, xline=85)

    data_cube = segyio.cube(read_segy_file)

    data_cube = np.swapaxes(data_cube, 1, 0)

    nptx, npty, nptz = data_cube.shape

    return data_cube, nptx, npty, nptz

#######################################################################################
# Creates receiver coordinates used in acquisition


def coor_generator(start, end, width, bin_xline, bin_inline):
    '''Creates rectangle shaped seismic data recording acqusition geometry'''
    # Note:
    # start - X,Y,Z coordinates(with respect to simulation model) in m where rectange should start
    # end - X,Y,Z coordinates(with respect to simulation model) in m where rectange should end
    # width - xline rectangle length(with respect to simulation model) in m
    # bin_xline - bin size in xline direction in m
    # bin_inline - bin size in inline direction in m

    # x_coor = np.linspace(start[0], end[0], int(
    #     np.abs(start[0]-end[0])/bin_inline))

    x_coor = np.arange(start[0]-width/2, start[0] + width/2, bin_xline)

    if len(x_coor) == 0:
        x_coor = start[0]

    # y_coor = np.linspace(start[1]-width/2, start[1] +
    #                      width/2, int(np.abs(width)/bin_xline))

    y_coor = np.arange(start[1], end[1], bin_inline)

    if len(y_coor) == 0:
        y_coor = start[1]

    z_coor = np.arange(start[2], end[2], bin_inline)

    if len(z_coor) == 0:
        z_coor = start[2]

    xx, yy, zz = np.meshgrid(x_coor, y_coor, z_coor)

    rec_coor = np.dstack([xx, yy, zz])

    rec_coor = rec_coor.transpose(2, 0, 1).reshape(3, -1)

    rec_coor = rec_coor.T

    return rec_coor


#######################################################################################
# Subsurface model properties 3D plotted


def plot_subs_model(model, grid_spacing, colorbar_label, cmap, figure_title, fig_name, save=False, visualize=False):
    '''Plot 3D subsurface models using PyVista python package'''
    # Note:
    # It is assumed model has origin (0,0,0).
    # Grid spacing is in m. Assumed uniform in all directions
    # Colorbar_label should be in quotes
    # any cmap for plotting, should be in quotes
    # Figure_title should be in quotes
    # save, either True or False
    # fig_name should in quotes
    # show, either True or False

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=model.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (grid_spacing, grid_spacing, grid_spacing)

    # Fill grid with data
    grid.point_data[colorbar_label] = model.flatten(order='F')

    # Start plotting velocity model
    if visualize:
        pl = pv.Plotter(title=figure_title)
    else:
        pl = pv.Plotter(title=figure_title, off_screen=True)

    pl.set_background('grey', top='black')
    
    # Set a custom position and size of colorbar
    # sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    pl.add_mesh(grid, show_edges=False, line_width=5,
                cmap=cmap, scalar_bar_args=sargs)

    # Add bounds and bound labels
    pl.show_bounds(grid, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, 14000, 0, 10000, 0, 4000])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    pl.camera_position = [(-12590.592718281714, -11962.645016328186, -3776.3167104272843),
                          (1172.83396018759, 395.40657698388264, 820.5129171311286),
                          (0.15457317302520832, 0.18859093540817437, -0.969814721100267)]

    # pl.camera_position = [(6997.5, 5002.5, -47957.43487814994),
    #  (6997.5, 5002.5, 2002.5),
    #  (1.0, 0.0, 0.0)]

    if visualize:
        pl.show()

    if save:
        save_name = fig_name
        pl.show(screenshot=save_name, window_size=[1280, 720])


def plot_subs_model_with_2Dline(model, TwoDline_coor, grid_spacing, colorbar_label, cmap, figure_title, fig_name, save=False, visualize=False):
    '''Plot 2D subsurface models using PyVista python package'''
    # Note:
    # It is assumed model has origin (0,0,0).
    # Grid spacing is in m. Assumed uniform in all directions
    # Colorbar_label should be in quotes
    # any cmap for plotting, should be in quotes
    # Figure_title should be in quotes
    # save, either True or False
    # fig_name should in quotes
    # show, either True or False

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=model.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (grid_spacing, grid_spacing, grid_spacing)

    # Fill grid with data
    grid.point_data[colorbar_label] = model.flatten(order='F')

    # Start plotting velocity model
    if visualize:
        pl = pv.Plotter(title=figure_title)
    else:
        pl = pv.Plotter(title=figure_title, off_screen=True)

    pl.set_background('grey', top='black')
    
    # Set a custom position and size of colorbar
    # sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    pl.add_mesh(grid, show_edges=False, line_width=5,
                cmap=cmap, scalar_bar_args=sargs)


    # Add 2D line
    TwoDline = np.array(TwoDline_coor)

    pl.add_points(TwoDline, color='white', point_size=5)

    legend_entries = []
    legend_entries.append(['Seismic line', 'white'])
    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')

    # Add bounds and bound labels
    pl.show_bounds(grid, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, 14000, 0, 10000, 0, 4000])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    pl.camera_position = [(-12590.592718281714, -11962.645016328186, -3776.3167104272843),
                          (1172.83396018759, 395.40657698388264, 820.5129171311286),
                          (0.15457317302520832, 0.18859093540817437, -0.969814721100267)]

    # pl.camera_position = [(6997.5, 5002.5, -47957.43487814994),
    #  (6997.5, 5002.5, 2002.5),
    #  (1.0, 0.0, 0.0)]
    if visualize:
        pl.show()

    if save:
        save_name = fig_name
        pl.show(screenshot=save_name, window_size=[1280, 720])
    

def plot_a_2Dline(model, dx, dy, dz, TwoDline_coor, figure_title, fig_name, cmap, visualize, save):
    #Find length of axis
    x_len = np.round(model.shape[0]*dx, -3)
    y_len = np.round(model.shape[1]*dy, -3)
    z_len = np.round(model.shape[-1]*dz, -3)

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=model.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)

    # Fill grid with data
    grid.point_data[figure_title] = model.flatten(order='F')

    # Start plotting model

    if visualize:
        pl = pv.Plotter(title=figure_title)
    else:
        pl = pv.Plotter(title=figure_title, off_screen=True)

    pl.set_background('grey', top='black')

    single_slice = grid.slice(normal='x', origin=[7000,0,0])

    # Set a custom position and size of colorbar
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    
    pl.add_mesh(single_slice, show_edges=False, line_width=5,
                cmap=cmap, scalar_bar_args=sargs)

    TwoDline = np.array(TwoDline_coor)

    pl.add_points(TwoDline, color='white', point_size=8)

    legend_entries = []
    legend_entries.append(['Seismic line', 'white'])
    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')
 

    # Add bounds and bound labels
    pl.show_bounds(single_slice, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, 14000, 0, 10000, 0, 4000])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')

    pl.camera_position = 'yz'
    pl.camera.roll -= 180
    pl.camera.azimuth = 180

    if visualize:
        pl.show()

    if save:
        save_name = fig_name
        pl.show(screenshot=save_name, window_size=[1280, 720])


def plot_a_slice(model, dx, dy, dz, src_coor, rec_coor, figure_title, cmap, save, visualize):
    #Find length of axis
    x_len = np.round(model.shape[0]*dx, -3)
    y_len = np.round(model.shape[1]*dy, -3)
    z_len = np.round(model.shape[-1]*dz, -3)

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=model.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)

    # Fill grid with data
    grid.point_data[figure_title] = model.flatten(order='F')

    # Start plotting wavefield model

    if visualize:
        pl = pv.Plotter(title=figure_title)
    else:
        pl = pv.Plotter(title=figure_title, off_screen=True)

    pl.set_background('grey', top='black')

    single_slice = grid.slice(normal='x', origin=[7000,0,0])

    # Set a custom position and size of colorbar
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    
    pl.add_mesh(single_slice, show_edges=False, line_width=5,
                cmap=cmap, scalar_bar_args=sargs)


    # Add source
    Point_labels = ['Source']
    source_point = np.array(src_coor)

    # Add receivers
    rec_points = rec_coor

    pl.add_points(source_point, color='red', point_size=12)
    pl.add_points(rec_points, color='green', point_size=6)

 
    # Add bounds and bound labels
    pl.show_bounds(single_slice, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, 14000, 0, 10000, 0, 4000])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    legend_entries = []
    legend_entries.append(['Source', 'red'])
    legend_entries.append(['Receivers', 'blue'])
    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')

    pl.camera_position = 'yz'
    pl.camera.roll -= 180
    pl.camera.azimuth = 180

    if visualize:
        pl.show()

    if save:
        save_name = f"figure_title"+'.png'
        pl.show(screenshot=save_name, window_size=[1280, 720])


def plot_subs_model_with_source(model, dx, dy, dz, src_coor, rec_coor, cmap, visualize, save):
    #Find length of axis
    x_len = np.round(model.shape[0]*dx, -3)
    y_len = np.round(model.shape[1]*dy, -3)
    z_len = np.round(model.shape[-1]*dz, -3)

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=model.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)

    # Fill grid with data
    grid.point_data["Bulk density, g/cm^3"] = model.flatten(order='F')

    # Start plotting wavefield model

    if visualize:
        pl = pv.Plotter(title=('Acoustic wave simulation'))
    else:
        pl = pv.Plotter(title=('Acoustic wave simulation'), off_screen=True)

    pl.set_background('grey', top='black')

    #orth_slices = grid.slice_orthogonal(x=x_len/2, y=y_len/2, z=None)

    # Set a custom position and size of colorbar
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    pl.add_mesh(grid, show_edges=False, line_width=5,
                cmap=cmap, scalar_bar_args=sargs)

    # Add source
    Point_labels = ['Source']
    source_point = np.array(src_coor)

    # Add receiver
    rec_point = rec_coor

    pl.add_points(source_point, color='red', point_size=12)
    pl.add_points(rec_point, color='green', point_size=6)

    # Add bounds and bound labels
    pl.show_bounds(grid, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, x_len, 0, y_len, 0, z_len])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    legend_entries = []
    legend_entries.append(['Source', 'red'])
    legend_entries.append(['Receivers', 'green'])
    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')

    pl.camera_position = [(-12590.592718281714, -11962.645016328186, -3776.3167104272843),
                          (1172.83396018759, 395.40657698388264,
                           820.5129171311286),
                          (0.15457317302520832, 0.18859093540817437, -0.969814721100267)]

    if visualize:
        pl.show()

    if save:
        save_name = 'Model_with_source'  + \
            (f"{src_coor[0]}")+'_'+(f"{src_coor[1]}")+'.png'
        pl.show(screenshot=save_name, window_size=[1280, 720])

#######################################################################################
#----------------------------- MAIN CODE STARTS HERE----------------------------------#
#######################################################################################
#Start counting simulation time
start = time.perf_counter()

# Load velocity and density cubes
vel_file = '/home/jokhongir/Modelling/Devito_with_density/Iversen2008_smooth_model_15m_grid.sgy'
velocity_cube, nptx, npty, nptz = read_segy_file(vel_file)


dens_file = '/home/jokhongir/Modelling/Devito_with_density/Iversen2008_smooth_model_15m_grid_density.sgy'
density_cube, nptx_dens, npty_dens, nptz_dens = read_segy_file(dens_file)

#-------------------------------------------------------------------------------------#
# Check cubes' dimensions

if nptx == nptx_dens and npty == npty_dens and nptz == nptz_dens:
    print('GOOD: Velocity and density cubes have equal dimensions')
else:
    print('WARNING: Velocity and density cubes are not equal')


#######################################################################################
# Plot subsurface model's properties
dx = 15  # m
dx = np.float32(dx)

dy = 15  # m
dy = np.float32(dy)

dz = 15  # m
dz = np.float32(dz)

#######################################################################################
#---------------------------DEFINE PARAMETRS FOR PLOTTING#---------------------------
#######################################################################################
shape = (nptx, npty, nptz)
spacing = (dx, dy, dz)  # m
origin = (0., 0., 0.)  # m

#-------------------------------------------------------------------------------------#
# Add source coor.
source_start = [6950, 1, 0]  # m
source_end = [6950, 5020, 0]  # m
source_width = 0  # m
source_int_x = 15  # m
source_int_y = 15  # m


source_coor_storage = coor_generator(
    start=source_start, end=source_end, width=source_width, bin_xline=source_int_x, bin_inline=source_int_y)

source_coor_storage = source_coor_storage.astype(np.float32)
#-------------------------------------------------------------------------------------#
# Add receivers coor.
recivers_coor_storage = []

first_rec_offset = 50  # m
last_rec_offset = 5000  # m

streamer_length = last_rec_offset - first_rec_offset  # m

for i in range(source_coor_storage.shape[0]):

    rec_start = [source_coor_storage[i][0]+100,
                 source_coor_storage[i][1]+first_rec_offset, source_coor_storage[i][2]]  # m
    rec_end = [source_coor_storage[i][0]+100, source_coor_storage[i]
               [1]+last_rec_offset, source_coor_storage[i][2]]  # m
    rec_width = 0  # m
    rec_int_x = 15  # m
    rec_int_y = 15  # m

    recivers_coor = coor_generator(
        start=rec_start, end=rec_end, width=rec_width, bin_xline=rec_int_x, bin_inline=rec_int_y)

    recivers_coor_storage.append(recivers_coor)

recivers_coor_storage = np.array(recivers_coor_storage)

recivers_coor_storage = recivers_coor_storage.astype(np.float32)


#######################################################################################
# Plot P-wave speed and bulk density models used for the simulation
# plot_subs_model(model=velocity_cube, grid_spacing=dx, colorbar_label="P-wave speed, km/s",
#                 cmap='jet', figure_title='Velocity model', fig_name='Velocity_model.png', save=True, visualize=False)

# # Plot density model
# plot_subs_model(model=density_cube, grid_spacing=dx, colorbar_label="Bulk density, g/cm^3",
#                 cmap='jet', figure_title='Density model', fig_name='Density_model.png', save=True, visualize=False)

# # Plot 2D line to be imaged  
# plot_subs_model_with_2Dline(model=density_cube, TwoDline_coor = source_coor_storage, grid_spacing=dx, colorbar_label="Bulk density, g/cm^3",
#                 cmap='jet', figure_title='Density model', fig_name='2D_seismic_line_in_model.png', save=True, visualize=False)

# plot_a_2Dline(model=density_cube, dx=dx, dy=dy, dz=dz, TwoDline_coor = source_coor_storage, figure_title='Bulk density, g/cm^3', fig_name='2D section',cmap='jet', save=True, visualize=False)

# #plot_a_slice(model=density_cube, dx=dx, dy=dy, dz=dz, src_coor=source_coor_storage[1], rec_coor=recivers_coor_storage[1], fig_title='Bulk density, g/cm^3', cmap='hsv', visualize=True, save=False)

for i in range(0, len(source_coor_storage),50):
  
    src_coor = source_coor_storage[i]
    rec_coor = recivers_coor_storage[i]

    plot_subs_model_with_source(model=density_cube, dx=dx, dy=dy, dz=dz, src_coor=src_coor, rec_coor=rec_coor, cmap = 'jet', visualize=False, save=True)
