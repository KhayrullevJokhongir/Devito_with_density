import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
from shapely.geometry import Point, LineString
import geopandas as gpd
import pandas as pd
from fiona.crs import from_epsg
import time
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


def bin_the_midpoints(bins, midpoints):
    b = bins.copy()
    m = midpoints.copy()
    reindexed = b.reset_index().rename(columns={'index': 'bins_index'})
    joined = gpd.tools.sjoin(reindexed, m)
    bin_stats = joined.groupby('bins_index')['offset'].aggregate(
        fold=len, min_offset='min')
    return gpd.GeoDataFrame(b.join(bin_stats))


def timer(start, CPU_number):
    end = time.perf_counter()
    #print(f'{end-start}')
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f'Calculations with {CPU_number} CPU took: \n{hours} hours, \n{minutes} minutes, \n{seconds} seconds')


# Start counting code running time
start = time.perf_counter()

#######################################################################################
# --------------------------------Create survey geometry--------------------------------
#######################################################################################
xmi = 0       # X coordinate of bottom-left corner of survey geometry (m)
ymi = 0       # Y coordinate of bottom-left corner of survey geometry (m)

x = 14000     # X extent of survey (m)
y = 10000     # Y extent of survey (m)


#-------------------------------------------------------------------------------------#
# Add source coor.
source_start = [7000, 0, 0]  # m
source_end = [7000, 400, 0]  # m
source_width = 0  # m
source_int_x = 15  # m
source_int_y = 15  # m


source_coor_storage1 = coor_generator(
    start=source_start, end=source_end, width=source_width, bin_xline=source_int_x, bin_inline=source_int_y)

source_start = [7000, 2000, 0]  # m
source_end = [7000, 8000, 0]  # m
source_width = 0  # m
source_int_x = 240  # m
source_int_y = 240  # m


# source_coor_storage2 = coor_generator(
#     start=source_start, end=source_end, width=source_width, bin_xline=source_int_x, bin_inline=source_int_y)


source_start = [7000, 9600, 0]  # m
source_end = [7000, 10020, 0]  # m
source_width = 0  # m
source_int_x = 15  # m
source_int_y = 15  # m


source_coor_storage3 = coor_generator(
    start=source_start, end=source_end, width=source_width, bin_xline=source_int_x, bin_inline=source_int_y)


source_coor_storage = np.vstack((source_coor_storage1, source_coor_storage3))

#-------------------------------------------------------------------------------------#
# Add receiver coor.
rec_start = [7000, 0, 0]  # m
rec_end = [7000, 10020, 0]  # m
rec_width = 0  # m
rec_int_x = 15  # m
rec_int_y = 15  # m

recivers_coor_storage = coor_generator(
    start=rec_start, end=rec_end, width=rec_width, bin_xline=rec_int_x, bin_inline=rec_int_y)


#######################################################################################
# Prepare data for further processing and plotting
rcvrs = [Point(x, y) for x, y in zip(
    recivers_coor_storage[:, 0], recivers_coor_storage[:, 1])]
srcs = [Point(x, y) for x, y in zip(
    source_coor_storage[:, 0], source_coor_storage[:, 1])]


#-------------------------------------------------------------------------------------#
station_list = ['Receiver']*len(rcvrs) + ['Source']*len(srcs)
survey = gpd.GeoDataFrame({'geometry': rcvrs+srcs, 'station': station_list})


try:
    # Needs geopandas fork: https://github.com/kwinkunks/geopandas
    survey.plot(figsize=(12, 12), column='station',
                cmap="bwr", markersize=2, legend=True)
except:
    # This will work regardless.
    survey.plot()
plt.grid()
plt.title('Sources and Receivers', fontsize=20)
plt.xlim(0, 14000)
plt.ylim(0, 10000)
plt.xlabel('x1, m', fontsize=15)
plt.ylabel('x2, m', fontsize=15)
plt.show()


#######################################################################################
# --------------------------------Calculate mid points----------------------------------
#######################################################################################
# Create survey ID for source and receivers
sid = np.arange(len(survey))
survey['SID'] = sid
# survey.to_file('survey_orig.shp')

midpoint_list = [LineString([r, s]).interpolate(0.5, normalized=True)
                 for r in rcvrs
                 for s in srcs]

offsets = [r.distance(s)
           for r in rcvrs
           for s in srcs]

azimuths = [np.arctan((r.x - s.x)/(r.y - s.y+1e-10))
            for r in rcvrs
            for s in srcs]

midpoints = gpd.GeoDataFrame({'geometry': midpoint_list,
                              'offset': offsets,
                              'azimuth': np.degrees(azimuths),
                              })

midpoints[:5]

#-------------------------------------------------------------------------------------#
# Plot midpoints
ax = midpoints.plot(figsize=(12, 12), markersize=2, legend=True)
plt.grid()
plt.title('Midpoints', fontsize=20)
plt.xlim(0, 14000)
plt.ylim(0, 10000)
plt.xlabel('x1, m', fontsize=15)
plt.ylabel('x2, m', fontsize=15)
plt.show()

# # midpoints.to_file('midpoints.shp')

# midpoints['offsetx'] = offsets * np.sin(azimuths)
# midpoints['offsety'] = offsets * np.cos(azimuths)
# midpoints[:5].offsetx  # Easy!

# x = [m.geometry.x for i, m in midpoints.iterrows()]
# y = [m.geometry.y for i, m in midpoints.iterrows()]

# fig = plt.figure(figsize=(12, 8))
# plt.quiver(x, y, midpoints.offsetx, midpoints.offsety, units='xy',
#            width=0.5, scale=1/0.025, pivot='mid', headlength=0)
# plt.axis('equal')
# plt.grid()
# plt.xlabel('x1, m')
# plt.ylabel('x2, m')
# plt.show()

#######################################################################################
# --------------------------------Calculate Bin coordinates-----------------------------
#######################################################################################
# Bin size in x and y directions
bin_x = 30  # m
bin_y = 30  # m

shift2_middle_x = bin_x / 2  # for creating bin coordinates
shift2_middle_y = bin_y / 2  # for creating bin coordinates

x_values = np.arange(bin_x, 14000, bin_x) - shift2_middle_x

#x_values = np.array([7500])
y_values = np.arange(bin_y, 10000, bin_y) - shift2_middle_y
x_mesh, y_mesh = np.meshgrid(x_values, y_values)

x_mesh = x_mesh.flatten()
y_mesh = y_mesh.flatten()

bin_centres = gpd.GeoSeries([Point(x, y) for x, y in zip(x_mesh, y_mesh)])

bin_polys = bin_centres.buffer((bin_x)/2, cap_style=3)
bins = gpd.GeoDataFrame(geometry=bin_polys)

bins[:3]
ax = bins.plot(figsize=(12, 12))
plt.title('Seismic bins', fontsize=20)
plt.xlabel('x1, m', fontsize=15)
plt.ylabel('x2, m', fontsize=15)
plt.xlim(0, 14000)
plt.ylim(0, 10000)
plt.show()

#######################################################################################
# Spatial join
#######################################################################################
bin_stats = bin_the_midpoints(bins, midpoints)
bin_stats[:10]

cbar_steps = round((bin_stats['fold'].max()-bin_stats['fold'].min())/60)*10
cbar_ticks = np.arange(bin_stats['fold'].min(
), bin_stats['fold'].max(), cbar_steps).tolist()
cbar_ticks = cbar_ticks[:-1]
cbar_ticks.append(bin_stats['fold'].max())

ax = bin_stats.plot(figsize=(12, 12), column="fold", cmap='jet', vmin=bin_stats['fold'].min(
), vmax=bin_stats['fold'].max(), legend=True, legend_kwds={'location': 'right', 'shrink': 0.58, 'ticks': cbar_ticks})

ax.grid(True)
plt.title('Seismic fold', fontsize=20)
plt.xlabel('x1, m', fontsize=15)
plt.ylabel('x2, m', fontsize=15)
plt.xlim(0, 14000)
plt.ylim(0, 10000)
plt.show()

# Calculate and print total simulation time
timer(start, 1)

print('\nSources spacing:')
print('on direction of x', source_int_x, 'm')
print('on direction of y', source_int_y, 'm')


print('\nReceivers spacing:')
print('on direction of x', rec_int_x, 'm')
print('on direction of y', rec_int_y, 'm')


print('\nA bin size:')
print('on direction of x', bin_x, 'm')
print('on direction of y', bin_x, 'm')


#Print()
