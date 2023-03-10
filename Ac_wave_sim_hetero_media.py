import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import concurrent.futures as cf
import pyvista as pv
import segyio

from devito import *
from scipy import interpolate


from examples.seismic.source import RickerSource, WaveletSource, TimeAxis
from examples.seismic import ModelViscoacoustic, plot_image, setup_geometry, plot_velocity
from examples.seismic import Model
from examples.seismic import Receiver


#######################################################################################
def timer(start, CPU_number):
    end = time.perf_counter()
    print(f'{end-start}')
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f'Simulation with {CPU_number} CPU took: \n{hours} hours, \n{minutes} minutes, \n{seconds} seconds')

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
#---------------------Functions for wave simulation using devito----------------------------
#######################################################################################
# Creates source and receivers for a stencil


def src_rec(p, model, src_coor, rec_coor):
    src = RickerSource(name='src', grid=model.grid,
                       f0=f0, time_range=time_range)
    src.coordinates.data[:] = src_coor

    # Create symbol for receivers
    rec = Receiver(name='rec', grid=model.grid,
                   npoint=rec_coor.shape[0], time_range=time_range)

    # Prescribe even spacing for receivers along the x-axis
    rec.coordinates.data[:] = rec_coor

    src_term = src.inject(field=p.forward, expr=(s**time_order*src))
    rec_term = rec.interpolate(expr=p)

    return src_term + rec_term, src, rec

#######################################################################################
# Creates stencil from Jan Thorbacke 2023


def create_stencil(model, p, v):

    # Bulk modulus
    bm = rho * (vp * vp)

    # Define PDE to v
    pde_v = v.dt + b * grad(p)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    # Define PDE to p
    pde_p = p.dt + bm * div(v.forward)
    u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

    return [u_v, u_p]

#######################################################################################
# Perfors sSismic Modelling using heterogenious acoustic wave equation.


def modelling_AWS(model, src_coor, rec_coor, shot_number):

    # Create symbols for particle velocity, pressure field, memory variable, source and receivers

    v = VectorTimeFunction(name="v", grid=model.grid,
                           time_order=time_order, space_order=space_order)

    p = TimeFunction(name="p", grid=model.grid, time_order=time_order, space_order=space_order,
                     staggered=NODE)

    # define the source injection and create interpolation expression for receivers

    src_rec_expr, src, rec = src_rec(p, model, src_coor, rec_coor)

    eqn = create_stencil(model, p, v)

    op = Operator(eqn + src_rec_expr, subs=model.spacing_map)

    op(time=time_range.num-1, dt=dt, src=src, rec=rec)

    trimmed_data = p.data[1, nbl:-nbl, nbl: -nbl, nbl: -nbl]

    return trimmed_data, rec, src_coor, rec_coor, shot_number

#######################################################################################
#--------------Functions for simulation using devito have ended------------------------
#######################################################################################

#######################################################################################
# Subsurface model properties 3D plotted


def plot_subs_model(model, grid_spacing, colorbar_label, cmap, figure_title, fig_name, save=False, show=False):
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
    pl = pv.Plotter(title=(figure_title))

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

    if save:
        return pl.save_graphic(fig_name)

    if show:
        return pl.show()

#######################################################################################
# Receivers recording 2D plotter


def plot_receiver(rec, src_coor, rec_coor, visualize, save):
    plt.figure(1)
    plt.imshow(rec.data, cmap='seismic', aspect='auto', vmax=0.005,
               vmin=-0.005, extent=(0, rec_coor.shape[0], tn, t0))
    plt.xlabel("Reciever number")
    plt.ylabel("Time (ms)")
    plt.title('Recordings at '+(f"{tn/1000}")+'s')
    plt.colorbar()

    if visualize:
        plt.show()

    if save:
        name = 'Recording_'+(f"{tn/1000}")+'s_shot' + \
            (f"{src_coor[0]}")+'_'+(f"{src_coor[1]}")+'.png'
        plt.savefig(name, dpi=200, bbox_inches='tight')

    plt.close()

#######################################################################################
# Plot 3 orthogonal slices of sim. data in 3D


def plot_sim_cube(trimmed_data, dx, dy, dz, src_coor, rec_save, visualize, save):
    #Find length of axis
    x_len = np.round(trimmed_data.shape[0]*dx, -3)
    y_len = np.round(trimmed_data.shape[1]*dy, -3)
    z_len = np.round(trimmed_data.shape[-1]*dz, -3)

    # Create grid for data to be plotted
    grid = pv.UniformGrid(dimensions=trimmed_data.shape)

    # Define origin and spacing between grids
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)

    # Fill grid with data
    grid.point_data["Acoustic pressure, Pa"] = trimmed_data.flatten(order='F')

    # Start plotting wavefield model

    if visualize:
        pl = pv.Plotter(title=('Acoustic wave simulation'))
    else:
        pl = pv.Plotter(title=('Acoustic wave simulation'), off_screen=True)

    pl.set_background('grey', top='black')

    orth_slices = grid.slice_orthogonal(x_len/2, y_len/2, z_len/2)

    # Set a custom position and size of colorbar
    sargs = dict(vertical=False, label_font_size=12, title_font_size=15)
    pl.add_mesh(orth_slices, show_edges=False, line_width=5,
                cmap='seismic', clim=[-0.005, 0.005], scalar_bar_args=sargs)

    # Add source
    Point_labels = ['Source']
    source_point = np.array(src_coor)

    # Add receiver
    rec_point = rec_save.coordinates.data

    pl.add_points(source_point, color='red', point_size=12)
    pl.add_points(rec_point, color='blue', point_size=6)

    # Add bounds and bound labels
    pl.show_bounds(orth_slices, zlabel='Depth, m', xlabel='X1, m',
                   ylabel='X2, m', axes_ranges=[0, x_len, 0, y_len, 0, z_len])

    labels = dict(zlabel='Depth, m', xlabel='X1, m', ylabel='X2, m')
    pl.add_axes(**labels)

    legend_entries = []
    legend_entries.append(['Source', 'red'])
    legend_entries.append(['Receivers', 'blue'])
    _ = pl.add_legend(legend_entries, size=(0.1, 0.1), face='rectangle')

    pl.camera_position = [(-12590.592718281714, -11962.645016328186, -3776.3167104272843),
                          (1172.83396018759, 395.40657698388264,
                           820.5129171311286),
                          (0.15457317302520832, 0.18859093540817437, -0.969814721100267)]

    if visualize:
        pl.show()

    if save:
        save_name = 'Wavefield_' + f"{tn/1000}"+'s_shot' + \
            (f"{src_coor[0]}")+'_'+(f"{src_coor[1]}")+'.png'
        pl.show(screenshot=save_name, window_size=[1280, 720])

#######################################################################################
# Time resampling for shot records


def resample(rec, num):
    start, stop = rec._time_range.start, rec._time_range.stop
    dt0 = rec._time_range.step

    new_time_range = TimeAxis(start=start, stop=stop, num=num)
    dt = new_time_range.step

    to_interp = np.asarray(rec.data)
    data = np.zeros((num, to_interp.shape[1]))

    for i in range(to_interp.shape[1]):
        tck = interpolate.splrep(
            rec._time_range.time_values, to_interp[:, i], k=3)
        data[:, i] = interpolate.splev(new_time_range.time_values, tck)

    coords_loc = np.asarray(rec.coordinates.data)
    # Return new object
    return data, coords_loc

#######################################################################################
# Segy writer for shot records


def segy_write(data, shot_number, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000):

    nt = data.shape[0]
    nsrc = 1
    nxrec = len(groupX)
    if sourceY is None and groupY is None:
        sourceY = np.zeros(1, dtype='int')
        groupY = np.zeros(nxrec, dtype='int')
    nyrec = len(groupY)

    # Create spec object
    spec = segyio.spec()
    spec.ilines = np.arange(nxrec)    # dummy trace count
    # assume coordinates are already vectorized for 3D
    spec.xlines = np.zeros(1, dtype='int')
    spec.samples = range(nt)
    spec.format = 1
    spec.sorting = 1

    with segyio.create(filename, spec) as segyfile:
        for i in range(nxrec):
            segyfile.header[i] = {
                segyio.su.tracl: i+1,
                segyio.su.tracr: i+1,
                segyio.su.fldr: shot_number,
                segyio.su.tracf: i+1,
                segyio.su.sx: round(np.round(sourceX[0] * np.abs(coordScalar))),
                segyio.su.sy: int(np.round(sourceY[0] * np.abs(coordScalar))),
                segyio.su.selev: int(np.round(sourceZ[0] * np.abs(elevScalar))),
                segyio.su.gx: int(np.round(groupX[i] * np.abs(coordScalar))),
                segyio.su.gy: int(np.round(groupY[i] * np.abs(coordScalar))),
                segyio.su.gelev: int(np.round(groupZ[i] * np.abs(elevScalar))),
                segyio.su.dt: int(dt*1e3),
                segyio.su.scalel: int(elevScalar),
                segyio.su.scalco: int(coordScalar)
                #segyio.su.iline: int(groupX[i]/dx+1),
                #segyio.su.xline: int(groupY[i]/dy+1),
                #segyio.su.cdpx: int((groupX[i]+sourceX[0])/2),
                #segyio.su.cdpy: int((groupY[i]+sourceY[0])/2),
                #segyio.su.offset: int(np.sqrt((sourceX[0]-groupX[i])**2+(sourceY[0]-groupY[i])**2))
            }
            segyfile.trace[i] = np.float32(data[:, i])
        segyfile.dt = int(dt*1e3)

#######################################################################################
# Saves a file written by segy_writer in .segy format


def save_rec(rec, shot_number, src_coords, recloc, nt, dt):

    if rec.data.size != 0:
        rec_save, coords = resample(rec, nt)

        segy_write(rec_save, shot_number,
                   [src_coords[0]],
                   [src_coords[-1]],
                   recloc[:, 0],
                   recloc[:, -1],
                   dt,  'Shot_'+(f"{src_coords[0]}") +
                   '_'+(f"{src_coords[1]}")+'.segy',
                   sourceY=[src_coords[1]],
                   groupY=recloc[:, 1],)


#######################################################################################
#----------------------------- MAIN CODE STARTS HERE----------------------------------#
#######################################################################################
#Start counting simulation time
start = time.perf_counter()

# Load velocity and density cubes
vel_file = 'Iversen2008_smooth_model_15m_grid.sgy'
velocity_cube, nptx, npty, nptz = read_segy_file(vel_file)


dens_file = 'Iversen2008_smooth_model_15m_grid_density.sgy'
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


# Plot P-wave speed and bulk density models used for the simulation
plot_subs_model(model=velocity_cube, grid_spacing=dx, colorbar_label="P-wave speed, km/s",
                cmap='jet', figure_title='Velocity model', fig_name=None, save=False, show=False)

# Plot density model
plot_subs_model(model=density_cube, grid_spacing=dx, colorbar_label="Bulk density, g/cm^3",
                cmap='jet', figure_title='Density model', fig_name=None, save=False, show=False)


#######################################################################################
#---------------------------DEFINE PARAMETRS FOR SIMULATION#---------------------------
#######################################################################################
shape = (nptx, npty, nptz)
spacing = (dx, dy, dz)  # m
origin = (0., 0., 0.)  # m
nbl = 50  # number of indexes in dumping layer
# order of numerical approximation used for calc. partial derivatives in space
space_order = 10
time_order = 2  # order of numerical approximation used for calc. partial derivatives in time
f0 = 0.015  # peak/dominant frequency in kHz
t0 = 0  # simulation start, ms
tn = 5000  # simulation end, ms

#-------------------------------------------------------------------------------------#

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
# Create the simulation model and time range for the simulation
model = Model(space_order=space_order, vp=velocity_cube, b=1/density_cube,
              origin=origin, shape=shape, spacing=spacing,
              nbl=nbl)

dt = np.trunc(model.critical_dt)  # ms
print(f'simulation time step is {dt} ms')
time_range = TimeAxis(start=t0, stop=tn, step=dt)

#######################################################################################
#Create paramters for generating devito's stencil
b = model.b  # bouyoncy/inverse of bulk density
rho = 1./b  # bulk density
vp = model.vp  # P-wave speed
lam = vp * vp * rho  # bulk modulus
damp = model.damp  # damping layer

s = model.grid.stepping_dim.spacing  # ds, time step in time

#######################################################################################
#------------------------PERFORM PARALELIZED SIMULATION-------------------------------#
#######################################################################################
source_coor_storage = source_coor_storage[::-1, :]
recivers_coor_storage = recivers_coor_storage[::-1, :, :]

simulation_storage = []
CPU_number = 12

for set in range(28):
    with cf.ProcessPoolExecutor(max_workers=CPU_number) as executor:
        if set == 27:
            for i in range(set*12,11+set*12): 
                src_coor = source_coor_storage[i]
                rec_coor = recivers_coor_storage[i]
                #######################################################################################
                #--------------------------SIMULATION TAKES PLACE HERE--------------------------------#
                #######################################################################################
                simulation = executor.submit(modelling_AWS, model, src_coor, rec_coor, shot_number=(i+1))
                simulation_storage.append(simulation)
        
        else:
            for i in range(set*12,12+set*12):
                src_coor = source_coor_storage[i]
                rec_coor = recivers_coor_storage[i] 
                #######################################################################################
                #--------------------------SIMULATION TAKES PLACE HERE--------------------------------#
                #######################################################################################
                simulation = executor.submit(modelling_AWS, model, src_coor, rec_coor, shot_number=(i+1))
                simulation_storage.append(simulation)
        
        for simulation in cf.as_completed(simulation_storage):
            
            trimmed_data, rec, src_coor_save, rec_coor_save, shot_number = simulation.result()
        
            #######################################################################################
            #--------------------PLOTTING AND SAVING PLOTS TAKE PLACE HERE------------------------#
            #######################################################################################
            # Plot (and save if desired) the simulation result in 3D
            plot_sim_cube(trimmed_data, dx, dy, dz, src_coor_save, rec,visualize=False, save=True)

            #######################################################################################
            # Plot (and save if desired) the receivers recording
            plot_receiver(rec, src_coor_save, rec_coor_save, visualize=False, save=True)

            #######################################################################################
            #-----------------CHECKING AND SAVING SIMULATION RESULTS TAKE PLACE HERE--------------#
            #######################################################################################
            # Check output, save simulation cube and data recorded on receivers
            info("Nan values : (%s, %s)" %
                (np.any(np.isnan(trimmed_data)), np.any(np.isnan(rec.data))))

            # info("Saving simulation cube")
            # np.save(f'Cube_of_shot_{src_coor_save}.npy',trimmed_data)
            # print(f'Simulation cube for shot {src_coor_save} is saved')

            info("Saving a shot records")
            save_rec(rec, shot_number, src_coor_save, rec_coor_save, tn, dt)
            print(f'Recordings of simulation {src_coor_save} is saved')

# Calculate and print total simulation time
timer(start, CPU_number)