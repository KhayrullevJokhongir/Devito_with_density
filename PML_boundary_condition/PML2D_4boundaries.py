# NBVAL_IGNORE_OUTPUT

import numpy                   as np
import matplotlib.pyplot       as plot
import math                    as mt
import matplotlib.ticker       as mticker    
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib              import cm

# NBVAL_IGNORE_OUTPUT

from   examples.seismic  import TimeAxis
from   examples.seismic  import RickerSource
from   examples.seismic  import Receiver
from   devito            import SubDomain, Grid, NODE, TimeFunction, Function, Eq, solve, Operator, div, grad


'''The mesh parameters that we choose define the domain plus the absorption region. For this, we use the following data:'''

nptx   =  101
nptz   =  101
x0     =  0.
x1     =  1000. 
compx  =  x1-x0
z0     =  0.
z1     =  1000.
compz  =  z1-z0;
hxv    =  (x1-x0)/(nptx-1)
hzv    =  (z1-z0)/(nptz-1)


'''The mesh parameters that we choose define the domain plus the absorption region. For this, we use the following data:'''
npmlx  = 20
npmlz  = 20



'''The lengths and are given, respectively, by:'''
lx = npmlx*hxv
lz = npmlz*hzv


'''We define the grid:'''

nptx   =  nptx + 2*npmlx
nptz   =  nptz + 2*npmlz
x0     =  x0 - hxv*npmlx
x1     =  x1 + hxv*npmlx
compx  =  x1-x0
z0     =  z0 - hzv*npmlz
z1     =  z1 + hzv*npmlz
compz  =  z1-z0
origin  = (x0,z0)
extent  = (compx,compz)
shape   = (nptx,nptz)
spacing = (hxv,hzv)

'''We use the structure of the subdomains to represent the white region and the blue region.

First, we define the white region, naming it as *d0'''

class d0domain(SubDomain):
    name = 'd0'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('middle', npmlx, npmlx), z: ('middle', npmlz, npmlz)}
d0_domain = d0domain()


'''The blue region is the union of 3 subdomains:'''

class d1domain(SubDomain):
    name = 'd1'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('left',npmlx), z: z}
d1_domain = d1domain()

class d2domain(SubDomain):
    name = 'd2'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('right',npmlx), z: z}
d2_domain = d2domain()

class d3domain(SubDomain):
    name = 'd3'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('middle', npmlx, npmlx), z: ('right',npmlz)}
d3_domain = d3domain()

class d4domain(SubDomain):
    name = 'd4'
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('middle', npmlx, npmlx), z: ('left',npmlz)}
d4_domain = d4domain()


'''The spatial grid is then defined:'''
grid = Grid(origin=origin, extent=extent, shape=shape, subdomains=(d0_domain,d1_domain,d2_domain,d3_domain,d4_domain), dtype=np.float64)


'''The velocity field is needed in both staggered and non-staggered grids. As before we, read the file and interpolate it to the non-staggered grid. From these values, we interpolate to the staggered grid.'''
v0 = np.zeros((nptx,nptz))
v1 = np.zeros((nptx-1,nptz-1))
X0 = np.linspace(x0,x1,nptx)
Z0 = np.linspace(z0,z1,nptz)
    
x10 = x0+lx
x11 = x1-lx
        
z10 = z0
z11 = z1 - lz

xm = 0.5*(x10+x11)
zm = 0.5*(z10+z11)
        
pxm = 0
pzm = 0
        
for i in range(0,nptx):
    if(X0[i]==xm): pxm = i
            
for j in range(0,nptz):
    if(Z0[j]==zm): pzm = j
            
p0 = 0    
p1 = pzm
p2 = nptz
v0[0:nptx,p0:p1] = 1.5
v0[0:nptx,p1:p2] = 2.5

p0 = 0    
p1 = pzm
p2 = nptz-1
v1[0:nptx-1,p0:p1] = 1.5
v1[0:nptx-1,p1:p2] = 2.5



####################################################################################################################
#Density
rho0 = np.zeros((nptx,nptz))
rho1 = np.zeros((nptx-1,nptz-1))

rho0[0:nptx,p0:p1] = 1
rho0[0:nptx,p1:p2] = 1

rho1[0:nptx-1,p0:p1] = 1
rho1[0:nptx-1,p1:p2] = 1

#Bulk modulus
K0 = v0**2*rho0
K1 = v1**2*rho1

#Bulk_mod_over_dens
K0_dev_rho0 = K0/rho0
K1_dev_rho1 = K1/rho1


####################################################################################################################

'''Previously we introduced the local variables x10,x11,z10,z11,xm,zm,pxm and pzm that help us to create a specific velocity field, where we consider the whole domain (including the absorpion region). 
Below we include a routine to plot the velocity field.'''

def graph2dvel(vel):
        plot.figure()
        plot.figure(figsize=(16,8))
        fscale =  1/10**(3)
        scale  = np.amax(vel[npmlx:-npmlx,0:-npmlz])
        extent = [fscale*(x0+lx),fscale*(x1-lx), fscale*(z1-lz), fscale*(z0)]
        fig = plot.imshow(np.transpose(vel[npmlx:-npmlx,0:-npmlz]), vmin=0.,vmax=scale, cmap=cm.seismic, extent=extent)
        plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
        plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
        plot.title('Velocity Profile')
        plot.grid()
        ax = plot.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plot.colorbar(fig, cax=cax, format='%.2e')
        cbar.set_label('Velocity [km/s]')
        plot.show()

# graph2dvel(K0)
# graph2dvel(rho0)

'''We then define the time properties:'''
t0    = 0.
tn    = 3000.   
CFL   = 0.4
vmax  = np.amax(v0) 
dtmax = np.float64((min(hxv,hzv)*CFL)/(vmax))
ntmax = int((tn-t0)/dtmax)+1
dt0   = np.float64((tn-t0)/ntmax)

time_range = TimeAxis(start=t0,stop=tn,num=ntmax+1)
nt         = time_range.num - 1

'''The symbolic values associated with the spatial and temporal grids that are used in the composition of the equations are given by:'''
(hx,hz) = grid.spacing_map  
(x, z)  = grid.dimensions     
t       = grid.stepping_dim
dt      = grid.stepping_dim.spacing

'''We set the parameters for the Ricker source:'''
f0      = 0.01
nsource = 1
xposf   = 0.5*(compx-2*npmlx*hxv)
zposf   = 2

src = RickerSource(name='src',grid=grid,f0=f0,npoint=nsource,time_range=time_range,staggered=NODE,dtype=np.float64)
src.coordinates.data[:, 0] = xposf
src.coordinates.data[:, 1] = zposf

#src.show()

'''For the receivers:'''
nrec   = nptx
nxpos  = np.linspace(x0,x1,nrec)
nzpos  = 2

rec = Receiver(name='rec',grid=grid,npoint=nrec,time_range=time_range,staggered=NODE,dtype=np.float64)
rec.coordinates.data[:, 0] = nxpos
rec.coordinates.data[:, 1] = nzpos

'''The displacement field u is allocated'''
u = TimeFunction(name="u",grid=grid,time_order=2,space_order=2,staggered=NODE,dtype=np.float64)

'''The auxiliary functions phi1, phi2  
 will be two fields of second order in time and space, which use points of type staggered.'''

phi1 = TimeFunction(name="phi1",grid=grid,time_order=2,space_order=2,staggered=(x,z),dtype=np.float64)
phi2 = TimeFunction(name="phi2",grid=grid,time_order=2,space_order=2,staggered=(x,z),dtype=np.float64)

'''We set the velocity on the non-staggered grid'''
vel0 = Function(name="vel0",grid=grid,space_order=2,staggered=NODE,dtype=np.float64)
vel0.data[:,:] = v0[:,:]

'''and on the staggered one. Notice that the field has one less point in each direction.'''
vel1 = Function(name="vel1", grid=grid,space_order=2,staggered=(x,z),dtype=np.float64)
vel1.data[0:nptx-1,0:nptz-1] = v1

vel1.data[nptx-1,0:nptz-1] = vel1.data[nptx-2,0:nptz-1]
vel1.data[0:nptx,nptz-1]   = vel1.data[0:nptx,nptz-2]


'''We set the velocity on the non-staggered grid'''
bulk_mod0 = Function(name="bulk_mod0",grid=grid,space_order=2,staggered=NODE,dtype=np.float64)
bulk_mod0.data[:,:] = K0[:,:]

'''and on the staggered one. Notice that the field has one less point in each direction.'''
bulk_mod1 = Function(name="bulk_mod1", grid=grid,space_order=2,staggered=(x,z),dtype=np.float64)
bulk_mod1.data[0:nptx-1,0:nptz-1] = K1

bulk_mod1.data[nptx-1,0:nptz-1] = bulk_mod1.data[nptx-2,0:nptz-1]
bulk_mod1.data[0:nptx,nptz-1]   = bulk_mod1.data[0:nptx,nptz-2]

####################################################################################################################
'''We set the density on the non-staggered grid'''
dens0_inv = Function(name="dens0_inv",grid=grid,space_order=2,staggered=NODE,dtype=np.float64)
dens0_inv.data[:] = 1/rho0[:]

'''and on the staggered one. Notice that the field has one less point in each direction.'''
dens1_inv = Function(name="dens1_inv", grid=grid,space_order=2,staggered=(x,z),dtype=np.float64)
dens1_inv.data[0:nptx-1,0:nptz-1] = 1/rho1

dens1_inv.data[nptx-1,0:nptz-1] = dens1_inv.data[nptx-2,0:nptz-1]
dens1_inv.data[0:nptx,nptz-1]   = dens1_inv.data[0:nptx,nptz-2]
####################################################################################################################

'''We set the source term and receivers'''
src_term = src.inject(field=u.forward,expr=src*dt**2)
rec_term = rec.interpolate(expr=u)

'''The next step is to create the structures that reproduce the functions ksi1, ksi2 and 
 and then assign these functions to fields in non-staggered and staggered grids.'''

'''In terms of program variables, we have the following definitions:'''
x0pml  = x0 + npmlx*hxv 
x1pml  = x1 - npmlx*hxv 
z0pml  = z0 + npmlz*hzv   
z1pml  = z1 - npmlz*hzv 

'''Having set the boundaries  we create a function fdamp, i=1 is ksi1 and i=2 is ksi2'''
def fdamp(x,z,i):
    
    quibar  = 0.173
          
    if(i==1):
        a = np.where(x<=x0pml,(np.abs(x-x0pml)/lx),np.where(x>=x1pml,(np.abs(x-x1pml)/lx),0.))
        fdamp = quibar*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
    if(i==2):
        a = np.where(z<=z0pml,(np.abs(z-z0pml)/lz),np.where(z>=z1pml,(np.abs(z-z1pml)/lz),0.))
        fdamp = quibar*(a-(1./(2.*np.pi))*np.sin(2.*np.pi*a))
      
    return fdamp

'''We created the damping function that represents ksi1 and ksi2, 
. We now define arrays with the damping function values on grid points (staggered and non-staggered): c

The arrays D01 and D02 are associated with points of type staggered and represent the functions ksi1 and ksi2, 
, respectively.

The arrays D11 and D12 are associated with points of type non-staggered and represent the functions ksi1 and ksi2, 
, respectively.'''

def generatemdamp():
    
    X0     = np.linspace(x0,x1,nptx)    
    Z0     = np.linspace(z0,z1,nptz)
    X0grid,Z0grid = np.meshgrid(X0,Z0)
    X1   = np.linspace((x0+0.5*hxv),(x1-0.5*hxv),nptx-1)
    Z1   = np.linspace((z0+0.5*hzv),(z1-0.5*hzv),nptz-1)
    X1grid,Z1grid = np.meshgrid(X1,Z1)
   
    D01 = np.zeros((nptx,nptz))
    D02 = np.zeros((nptx,nptz))
    D11 = np.zeros((nptx,nptz))
    D12 = np.zeros((nptx,nptz))
    
    D01 = np.transpose(fdamp(X0grid,Z0grid,1))
    D02 = np.transpose(fdamp(X0grid,Z0grid,2))
  
    D11 = np.transpose(fdamp(X1grid,Z1grid,1))
    D12 = np.transpose(fdamp(X1grid,Z1grid,2))
    
    return D01, D02, D11, D12

D01, D02, D11, D12 = generatemdamp()

'''Below we include a routine to plot the damping fields.'''
def graph2damp(D):     
    plot.figure()
    plot.figure(figsize=(16,8))
    fscale = 1/10**(-3)
    fscale = 10**(-3)
    scale  = np.amax(D)
    extent = [fscale*x0,fscale*x1, fscale*z1, fscale*z0]
    fig = plot.imshow(np.transpose(D), vmin=0.,vmax=scale, cmap=cm.seismic, extent=extent)
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.title('Absorbing Layer Function')
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.set_label('Damping')
    plot.show()

    
# graph2damp(D01)
# graph2damp(D02)

#graph2damp(D11-D12)
# graph2damp(D12)

'''As pointed out previously, the functions ksi1 and ksi2 define damping in the directions x and z 
 respectively. They will be identified with the symbolic names of dampx and dampz, respectively.

As damping acts on non-staggered and staggered grids, we will identify dampx0 and dampz0 as being damping on the non-staggered points grid. 
Similarly, we will identify dampx1 and dampz1 as being the damping on the staggered points grid.'''

dampx0 = Function(name="dampx0", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
dampz0 = Function(name="dampz0", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
dampx0.data[:,:] = D01
dampz0.data[:,:] = D02

dampx1 = Function(name="dampx1", grid=grid,space_order=2,staggered=(x,z),dtype=np.float64)
dampz1 = Function(name="dampz1", grid=grid,space_order=2,staggered=(x,z),dtype=np.float64)
dampx1.data[0:nptx-1,0:nptz-1] = D11
dampz1.data[0:nptx-1,0:nptz-1] = D12

'''In terms of dimensions, the arrays D11 and D12 have dimension (nptx-1)*(nptz-1). As our grid has nptx*nptx 
 points, so we complete the line nptx-1 with information from the line nptx-2 and the column nptz-1 with information from the column nptz-2,
   in fields dampx1 and dampz1 using the arrays D11 and D12, respectively.'''

dampx1.data[nptx-1,0:nptz-1]   = dampx1.data[nptx-2,0:nptz-1]
dampx1.data[0:nptx,nptz-1]     = dampx1.data[0:nptx,nptz-2]
dampz1.data[nptx-1,0:nptz-1]   = dampz1.data[nptx-2,0:nptz-1]
dampz1.data[0:nptx,nptz-1]     = dampz1.data[0:nptx,nptz-2]

'''As we saw previously, the acoustic equation with PML has the formulations

In the white (interior) region:

eq1 = u.dt2 - vel0 * vel0 * u.laplace;
And in the blue (absorption) region:

eq2 = u.dt2 + (dampx0+dampz0) * u.dtc + (dampx0 * dampz0) * u - u.laplace * vel0 * vel0 + 
phi1[t,x,z] + phi2[t,x,z];

eq3 = phi1.dt + dampx1 * 0.5 * (phi1.forward+phi1) -(dampz1-dampx1) * u[t,x,z] * vel1 * vel1

eq4 = phi2.dt + dampz1 * 0.5 * (phi2.forward+phi2) -(dampx1-dampz1) * u[t,x,z] * vel1 * vel1

In the equation eq2 the term phi1[t,x,z] is given by following expression:
-(0.5/hx) * (phi1[t,x,z-1]+phi1[t,x,z]-phi1[t,x-1,z-1]-phi1[t,x-1,z]);

And the term phi2[t,x,z] in the equation eq2 is given by:
-(0.5/hz) * (phi2[t,x-1,z]+phi2[t,x,z]-phi2[t,x-1,z-1]-phi2[t,x,z-1]);

In the equation eq3 the term u[t,x,z] is given by:

a1 = u[t+1,x+1,z] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x,z+1];
a2 = u[t,x+1,z] + u[t,x+1,z+1] - u[t,x,z] - u[t,x,z+1];
u[t,x,z] = 0.5 * (0.5/hx) * (a1+a2);

In the equation eq4 the term u[t,x,z] is given by:
b1 = u[t+1,x,z+1] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x+1,z];
b2 = u[t,x,z+1] + u[t,x+1,z+1] - u[t,x,z] - u[t,x+1,z];
u[t,x,z] = 0.5 * (0.5/hz) * (b1+b2)

Then, using the operator Eq(eq) and the equation in the format associated with Devito 
we create the pdes that represent the acoustic equations with PML without the external force term in the white and blue regions, respectively by:'''
# White Region


#pde01   = Eq(u.dt2-bulk_mod0*(dens0_inv.dx+dens0_inv.dy)*(u.dx+u.dy)-vel0**2*u.laplace) 
pde01   = Eq(u.dt2-bulk_mod0*((dens0_inv.dx*u.dx+dens0_inv*u.dx2)+(dens0_inv.dy*u.dy+dens0_inv*u.dy2))) 


# Blue Region
#pde02a  = u.dt2   + (dampx0+dampz0)*u.dtc + (dampx0*dampz0)*u - bulk_mod0*(dens0_inv.dx+dens0_inv.dy)*(u.dx+u.dy)-vel0**2*u.laplace 

pde02a  = u.dt2   + (dampx0+dampz0)*u.dt + (dampx0*dampz0)*u -bulk_mod0*((dens0_inv.dx*u.dx+dens0_inv*u.dx2)+(dens0_inv.dy*u.dy+dens0_inv*u.dy2))


pde02b  = - (0.5/hx)*(phi1[t,x,z-1]+phi1[t,x,z]-phi1[t,x-1,z-1]-phi1[t,x-1,z])
pde02c  = - (0.5/hz)*(phi2[t,x-1,z]+phi2[t,x,z]-phi2[t,x-1,z-1]-phi2[t,x,z-1])
pde02   = Eq(pde02a + pde02b + pde02c)

pde10 = phi1.dt + dampx1*0.5*(phi1.forward+phi1)
a1    = u[t+1,x+1,z] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x,z+1] 
a2    = u[t,x+1,z]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x,z+1] 
pde11 = -(dampz1-dampx1)*0.5*(0.5/hx)*(a1+a2)*vel1**2
pde1  = Eq(pde10+pde11)
                                                    
pde20 = phi2.dt + dampz1*0.5*(phi2.forward+phi2) 
b1    = u[t+1,x,z+1] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x+1,z] 
b2    = u[t,x,z+1]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x+1,z] 
pde21 = -(dampx1-dampz1)*0.5*(0.5/hz)*(b1+b2)*vel1**2
pde2  = Eq(pde20+pde21)

                     
'''Now we define the stencils for each of the pdes that we created previously. The pde01 is defined on subdomain d0'''
stencil01 =  Eq(u.forward,solve(pde01,u.forward) ,subdomain = grid.subdomains['d0'])

'''The pdes: pde02, pde1 and pde2 are defined on the union of the subdomains d1, d2 and d3.'''
subds = ['d1','d2','d3','d4']

stencil02 = [Eq(u.forward,solve(pde02, u.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

stencil1 = [Eq(phi1.forward, solve(pde1,phi1.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

stencil2 = [Eq(phi2.forward, solve(pde2,phi2.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

'''The boundary conditions are set'''
bc  = [Eq(u[t+1,0,z],0.),Eq(u[t+1,nptx-1,z],0.),Eq(u[t+1,x,nptz-1],0.),Eq(u[t+1,x,0],0)]

'''We then define the operator (op) that will join the acoustic equation, source term, boundary conditions and receivers.

The acoustic wave equation in the d0 region: [stencil01];
The acoustic wave equation in the d1, d2 and d3 regions: [stencil02];
Source term: src_term;
Boundary Condition: bc;
Auxiliary function 
 in the d1, d2 and d3 regions: [stencil1];
Auxiliary function 
 in the d1, d2 and d3 regions: [stencil2];
Receivers: rec_term;
We then define the following op:'''

#op = Operator([stencil01,stencil02] + src_term + bc + [stencil1,stencil2] + rec_term,subs=grid.spacing_map)


op = Operator([stencil01,stencil02] + src_term + bc + [stencil1,stencil2] + rec_term,subs=grid.spacing_map)


'''So that there are no residuals in the variables of interest, we reset the fields u, phi1 and phi2 as follows:'''
u.data[:]     = 0.
phi1.data[:]  = 0.
phi2.data[:]  = 0.

'''We assign to op the number of time steps it must execute and the size of the time step in the local variables time and dt, respectively. 
This assignment is done as in Introduction to Acoustic Problem, where we have the following attribution structure:'''
op(time=nt,dt=dt0)

'''We view the result of the displacement field at the end time using the graph2d routine given by:'''
def graph2d(U):    
    plot.figure()
    plot.figure(figsize=(16,8))
    fscale =  1/10**(3)
    scale  = np.amax(U[npmlx:-npmlx,0:-npmlz])/10.
    extent = [fscale*x0pml,fscale*x1pml,fscale*z1pml,fscale*z0pml]
    fig = plot.imshow(np.transpose(U[npmlx:-npmlx,0:-npmlz]),vmin=-scale, vmax=scale, cmap=cm.seismic, extent=extent)
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
    plot.axis('equal')
    plot.title('Map - Acoustic Problem PML Devito')
    plot.grid()
    ax = plot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plot.colorbar(fig, cax=cax, format='%.2e')
    cbar.set_label('Displacement [km]')
    plot.draw()
    plot.show()

graph2d(u.data[0,:,:])

'''The solution obtained here has a reduction in the noise when compared with the results displayed in the notebook 
Introduction to Acoustic Problem. We plot the result of the Receivers using the graph2drec routine.'''

def graph2drec(rec):    
        plot.figure()
        plot.figure(figsize=(16,8))
        fscaled = 1/10**(3)
        fscalet = 1/10**(3)
        scale   = np.amax(rec[:,npmlx:-npmlx])/10.
        extent  = [fscaled*x0pml,fscaled*x1pml, fscalet*tn, fscalet*t0]
        fig = plot.imshow(rec[:,npmlx:-npmlx], vmin=-scale, vmax=scale, cmap=cm.seismic, extent=extent)
        plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f km'))
        plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
        plot.axis('equal')
        plot.title('Receivers Signal Profile with PML - Devito')
        ax = plot.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plot.colorbar(fig, cax=cax, format='%.2e')
        plot.show()

graph2drec(rec.data)
