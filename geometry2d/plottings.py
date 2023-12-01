from enum import Enum                     # for 2D vs 2D plots
import numpy as np                        # scientific computing tools

import matplotlib.pyplot as plt           # for plots
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D   # for 3D plots
plt.rcParams.update({"text.usetex":True, "font.family":"sans-serif", "font.sans-serif":["Helvetica"]}) # font properties

# Kind of coordinates
class Coords(Enum):
    PLANE=2   # 2D
    SPHERE=3  # 3D
    
# z orders
zc = 100
z_order_sphere     = zc; zc = zc+1
z_order_conj_surf  = zc; zc = zc+1
z_order_axes       = zc; zc = zc+1
z_order_geodesics  = zc; zc = zc+1
z_order_splitting  = zc; zc = zc+1
z_order_wavefront  = zc; zc = zc+1
z_order_conjugate  = zc; zc = zc+1
z_order_q0         = zc; zc = zc+1
delta_zo_back = 50

# Parameters for the 3D view
elevation__ = -10
azimuth__ = 20
dist__ = 10

#
alpha_sphere = 0.4

# 2D to 3D coordinates
def coord3d(theta, phi, epsilon):
    v = theta
    u = phi
    coefs = (1., 1., epsilon)                   # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
    rx, ry, rz = coefs                          # Radii corresponding to the coefficients
    x = rx * np.multiply(np.cos(u), np.cos(v))
    y = ry * np.multiply(np.cos(u), np.sin(v))
    z = rz * np.sin(u)
    return x, y, z

def get_cam(elev, azimuth, dist):
    ce   = np.cos(2*np.pi*elev/360)
    se   = np.sin(2*np.pi*elev/360)
    ca   = np.cos(2*np.pi*azimuth/360)
    sa   = np.sin(2*np.pi*azimuth/360)
    cam  = np.array([ dist*ca*ce, dist*sa*ce, dist*se])
    return cam

def plot3d(ax, x, y, z, azimuth, color, linewidth, linestyle='solid', zorder=1):
    N = len(x)
    i = 0
    j = 1
    cam = get_cam(elevation__, azimuth, dist__)
    ps = x[0]*cam[0]+y[0]*cam[1]+z[0]*cam[2]
    while i<N-1:
        ps_j = x[j]*cam[0]+y[j]*cam[1]+z[j]*cam[2]
        if (ps*ps_j<0) or (j==N-1):
            if ps>0:
                ls = linestyle
                lw = linewidth/3.0
                al = 0.5
                zo = zorder - delta_zo_back
            else:
                ls = linestyle
                lw = linewidth
                al = 1.0
                zo = zorder
            ax.plot(x[i:j+1], y[i:j+1], z[i:j+1], color=color, \
                    linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
            i = j
            ps = ps_j
        j = j+1
    
def decorate_2d(ax, q0=None):
    
    x   = [-np.pi, np.pi, np.pi, -np.pi]
    y   = [np.pi/2, np.pi/2, np.pi/2+1, np.pi/2+1]
    ax.fill(x, y, color=(0.95, 0.95, 0.95))
    y   = [-(np.pi/2+1), -(np.pi/2+1), -np.pi/2, -np.pi/2]
    ax.fill(x, y, color=(0.95, 0.95, 0.95))
    ax.set_xlabel(r'$\theta$', fontsize=10)
    ax.set_ylabel(r'$\varphi$', fontsize=10)
    ax.axvline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    ax.axhline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    ax.axhline(-np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
    ax.axhline( np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_aspect('equal', 'box')

    if not(q0 is None):
        r = q0[0]
        θ = q0[1]
        φ = r - np.pi/2
        ax.plot(θ, φ, marker="o", markersize=2, 
                markeredgecolor="black", markerfacecolor="black", zorder=z_order_q0)  
        
def init_figure_2d(q0=None):
    
    fig = Figure(dpi=200)
    fig.set_figwidth(3)
    fig.patch.set_alpha(0.0)
    ax  = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    decorate_2d(ax, q0)
    return fig

def plot_2d(fig, θ, φ, *, color='b', linewidth=0.5, zorder=1):
    ax = fig.axes[0]
    ax.plot(θ, φ, color=color, linewidth=linewidth, zorder=zorder)

def decorate_3d(ax, epsilon, q0=None, elevation=elevation__, azimuth=azimuth__):
    
    ax.set_axis_off()
    coefs = (1., 1., epsilon)              # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
    rx, ry, rz = coefs                     # Radii corresponding to the coefficients

    # Set of all spherical angles:
    v = np.linspace(-np.pi, np.pi, 100)
    u = np.linspace(-np.pi/2, np.pi/2, 100)

    # Cartesian coordinates that correspond to the spherical angles
    x = rx * np.outer(np.cos(u), np.cos(v))
    y = ry * np.outer(np.cos(u), np.sin(v))
    z = rz * np.outer(np.sin(u), np.ones_like(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, \
                    color=(0.99, 0.99, 0.99), alpha=alpha_sphere, \
                    antialiased=True, zorder=z_order_sphere)

    # initial point
    if not(q0 is None):
        r = q0[0]
        θ = q0[1]
        φ = r - np.pi/2
        x, y, z = coord3d(θ, φ, epsilon)
        cam = get_cam(elevation, azimuth, dist__)
        ps = x*cam[0]+y*cam[1]+z*cam[2]
        if ps>0: # back
            zo = z_order_q0 - delta_zo_back
            al = 0.5
        else:
            zo = z_order_q0
            al = 1.0
        ax.plot(x, y, z, marker="o", markersize=3, alpha=al, \
                markeredgecolor="black", markerfacecolor="black", zorder=zo)

    # add one meridian
    N = 100
    if not(q0 is None):
        θ = q0[1]*np.ones(N)
    else:
        θ = 0*np.ones(N)
    φ = np.linspace(0, 2*np.pi, N)
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, azimuth, color="black", \
            linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    # add equator
    N = 100
    θ = np.linspace(0, 2*np.pi, N)
    φ = 0*np.ones(N)
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, azimuth, color="black", \
            linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    
    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    ax.view_init(elev=elevation, azim=azimuth) # Reproduce view
    #ax.dist = dist__
    ax.set_box_aspect(None, zoom=dist__)
    
    ax.set_xlim(np.array([-rx,rx])*.67)
    ax.set_ylim(np.array([-ry,ry])*.67)
    ax.set_zlim(np.array([-rz,rz])*.67)

    # 
    ax.set_aspect('equal', 'box')   

def init_figure_3d(epsilon, q0=None, elevation=elevation__, azimuth=azimuth__):
    
    fig = Figure(dpi=200)
    fig.set_figwidth(2)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    decorate_3d(ax, epsilon, q0, elevation, azimuth)
    return fig

def plot_3d(fig, x, y, z, *, 
            color='b', 
            linewidth=1, 
            zorder=1):
    #
    ax = fig.axes[0]
    # get azimuth and elevation
    azimuth   = ax.azim
    elevation = ax.elev
    #
    N = len(x)
    i = 0
    j = 1
    #
    cam = get_cam(elevation, azimuth, dist__)
    ps = x[0]*cam[0]+y[0]*cam[1]+z[0]*cam[2]
    #
    while i<N-1:
        ps_j = x[j]*cam[0]+y[j]*cam[1]+z[j]*cam[2]
        if (ps*ps_j<0) or (j==N-1):
            if ps>0:
                ls = 'solid'
                lw = linewidth/3.0
                al = 0.5
                zo = zorder - delta_zo_back
            else:
                ls = 'solid'
                lw = linewidth
                al = 1.0
                zo = zorder
            ax.plot(x[i:j+1], y[i:j+1], z[i:j+1], color=color, \
                    linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
            i = j
            ps = ps_j
        j = j+1

# get surface coordinates from a closed curve defined by spherical coordinates
def surface_from_spherical_curve(x, y, epsilon):
    N = 100
    xmin = np.min(x)
    xmax = np.max(x)
    
    X = np.zeros((N, N))
    Y = np.zeros((N, N))
    
    xs = np.linspace(xmin, xmax, N)
    for i in range(N):
        x_current = xs[i]
        # find the two intersections of the curve with x_current
        ii  = np.argwhere(np.multiply(x[1:]-x_current, \
                                      x[0:-1]-x_current)<=0)
        #
        k   = ii[0][0]
        xk  = x[k]
        xkp = x[k+1]
        λ   = (x_current-xk)/(xkp-xk)
        y1  = y[k]+λ*(y[k+1]-y[k])
        #
        k   = ii[1][0]
        xk  = x[k]
        xkp = x[k+1]
        if abs(xkp-xk)>1e-12:
            λ = (x_current-xk)/(xkp-xk)
        else:
            λ = 0
        y2  = y[k]+λ*(y[k+1]-y[k])
        #
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        ys = np.linspace(ymin, ymax, N)
        X[:, i] = x_current*np.ones(N)
        Y[:, i] = ys
    
    # cartesian
    XX = np.zeros((N, N))
    YY = np.zeros((N, N))
    ZZ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            XX[i,j], YY[i,j], ZZ[i,j] = coord3d(X[i,j], Y[i,j], \
                                                epsilon)

    return XX, YY, ZZ
        
def plot_surface(fig, X, Y, Z, *, color='b', alpha=0.5, zorder=1):
    #
    ax = fig.axes[0]
    ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, \
                color=color, alpha=alpha, antialiased=True, zorder=zorder)