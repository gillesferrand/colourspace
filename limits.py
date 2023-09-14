# -*- coding: utf-8 -*-
"""
Find the MacAdam limits = optimal colour stimuli in CIE spaces
(c.f. http://www.brucelindbloom.com/index.html?LabGamutDisplayHelp.html)
Relies on the colour-science package for computing a colour from a spectral distribution
Relies on the scipy.spatial package for triangulating the boundary
"""

from __future__ import print_function
import sys
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import colour  # http://colour-science.org
    colour.utilities.filter_warnings()
except:
    #print("colour-science package not found")
    pass
from scipy.spatial import Delaunay

import os
this_dir, this_filename = os.path.split(__file__)
this_dir += "/limits"

#-----------------
# find the limits
#-----------------

limits_domain = {}
limits = {}
limits['cmp'] = {'XYZ': np.array([]), 'Lab': np.array([]), 'LCH': np.array([]), 'RGB': np.array([])}
limits['ref'] = {}

def get_limits_ref():
    """ Gets the limits tabulated inside colour-science, for reference """
    if hasattr(colour.volume,'ILLUMINANTS_OPTIMAL_COLOUR_STIMULI'): 
        limits['ref']['XYZ'] = colour.xyY_to_XYZ(colour.volume.ILLUMINANTS_OPTIMAL_COLOUR_STIMULI['D65'])
    if hasattr(colour.volume,'ILLUMINANT_OPTIMAL_COLOUR_STIMULI'): 
        limits['ref']['XYZ'] = colour.xyY_to_XYZ(colour.volume.ILLUMINANT_OPTIMAL_COLOUR_STIMULI['D65'])
    if hasattr(colour.volume,'OPTIMAL_COLOUR_STIMULI_ILLUMINANTS'): 
        limits['ref']['XYZ'] = colour.xyY_to_XYZ(colour.volume.OPTIMAL_COLOUR_STIMULI_ILLUMINANTS['D65'])
    if 'XYZ' in limits['ref'].keys():
        convert(limits['ref'])
        for space in limits['ref'].keys(): triangulate(space,kind='ref')

def find_limits(l_step=1, l_min=360, l_max=780, dir=this_dir):
    """ Computes the optimal colours
        l = wavelength in nanometers
        (allowed l_step = 20, 10, 5, 1)
    """
    global limits, limits_domain
    domain = np.arange(l_min,l_max+l_step,l_step)
    n = (len(domain)-1) * len(domain) + 2
    limits['cmp']['XYZ'] = np.zeros((n,3))
    limits_domain = {'l_min': l_min, 'l_max':l_max, 'l_step':l_step}
    # loop over all square pulses
    n = 0
    printline("i_width = %4i/%4i, i_start = %4i/%4i"%(0,len(domain),0,0), newline=False)
    values = np.zeros(len(domain))
    limits['cmp']['XYZ'][n] = spectrum_to_XYZ(domain, values)
    for i_width in range(1,len(domain)):
        for i_start in range(0,len(domain)):
            n += 1
            printline("i_width = %4i/%4i, i_start = %4i/%4i"%(i_width,len(domain),i_start,len(domain)-1), newline=False)
            values = np.zeros(len(domain))
            for i in np.arange(i_start, i_start+i_width, 1): values[i%len(domain)] = 1
            limits['cmp']['XYZ'][n] = spectrum_to_XYZ(domain, values)
    n += 1
    printline("i_width = %4i/%4i, i_start = %4i/%4i"%(len(domain),len(domain),0,0), newline=False)
    values = np.ones(len(domain))
    limits['cmp']['XYZ'][n] = spectrum_to_XYZ(domain, values)
    printline("n = %i                                 "%(n+1), newline=True)
    print_range(kind='cmp',space='XYZ')
    convert(limits['cmp'])
    #for space in limits['cmp'].keys(): triangulate(space,kind='cmp') # too slow to be done by default
    if dir != "": save_limits(dir=dir)

def printline(text, newline=False):
    sys.stdout.write(text)
    sys.stdout.write("\n") if newline else sys.stdout.write("\r")
    sys.stdout.flush()

def spectrum_to_XYZ(domain, values, illuminant='D65'):
    """ Converts a spectral distribution into a XYZ colour """
    data = dict(zip(domain,values))
    if hasattr(colour,'SpectralPowerDistribution'):
        spd = colour.SpectralPowerDistribution(data, interpolator=colour.NullInterpolator)
        cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        illuminant = colour.ILLUMINANTS_RELATIVE_SPDS[illuminant]
        XYZ = colour.spectral_to_XYZ(spd, cmfs, illuminant)
    if hasattr(colour,'SpectralDistribution'):
        sd = colour.SpectralDistribution(data, interpolator=colour.NullInterpolator)
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        illuminant = colour.SDS_ILLUMINANTS[illuminant]
        XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
    return XYZ

def convert(dict, illuminant='D65'):
    """ Converts the set of XYZ colours into other spaces: Lab, LCH, sRGB """
    if hasattr(colour,'ILLUMINANTS'    ): illuminant = colour.ILLUMINANTS    ['CIE 1931 2 Degree Standard Observer'][illuminant]
    if hasattr(colour,'CCS_ILLUMINANTS'): illuminant = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
    dict['Lab'] = colour.XYZ_to_Lab(dict['XYZ']/100., illuminant) # beware: expects XYZ in [0,1]
    dict['LCH'] = colour.Lab_to_LCHab(dict['Lab'])
    dict['RGB'] = colour.XYZ_to_sRGB(dict['XYZ']/100., illuminant) # beware: expects XYZ in [0,1]

def print_range(kind='cmp',space=''):
    """ Prints the range of each coordinate """
    if kind not in limits.keys() or space not in limits[kind].keys() or len(limits[kind][space]) == 0: return
    for dim in range(3):
        print(space[dim]," = ",limits[kind][space][:,dim].min()," – ",limits[kind][space][:,dim].max())

def get_extremum(kind='cmp'):
    """ Gets the LCH value of the colour of highest C"""
    space = 'LCH'
    if kind not in limits.keys() or space not in limits[kind].keys() or len(limits[kind][space]) == 0: return
    i_max = limits[kind][space][:,1].argmax()
    return limits[kind][space][i_max]

#----------------------
# save/load the limits
#----------------------

def save_limits(dir=this_dir):
    """ Saves the list of optimal colours as a numpy binary file """
    global limits, limits_domain
    space = 'XYZ'
    l_min  = limits_domain['l_min']
    l_max  = limits_domain['l_max']
    l_step = limits_domain['l_step']
    fname = '%s/limits_%i-%i-%i_%s.npy'%(dir,l_min,l_max,l_step,space)
    print("saving limits to %s"%fname,end='')
    np.save(fname, limits['cmp'][space])
    print(": %i points"%(len(limits['cmp'][space])))

def load_limits(l_step, l_min=360, l_max=780, dir=this_dir):
    """ Loads the list of optimal colours from a numpy binary file """
    global limits, limits_domain
    space = 'XYZ'
    fname = '%s/limits_%i-%i-%i_%s.npy'%(dir,l_min,l_max,l_step,space)
    print("loading limits from %s"%fname,end='')
    limits['cmp'][space] = np.load(fname)
    print(": %i points"%(len(limits['cmp'][space])))
    limits_domain['l_min' ] = l_min
    limits_domain['l_max' ] = l_max
    limits_domain['l_step'] = l_step
    convert(limits['cmp'])
    #for space in limits['cmp'].keys(): triangulate(space,kind='cmp')

#-----------------
# plot the limits
#-----------------

index = {} # index of x, y, z axis
index['XYZ'] = [2, 0, 1]
index['Lab'] = [1, 2, 0]
index['LCH'] = [1, 2, 0]
index['RGB'] = [0, 1, 2]

def plot_limits3D(kind='cmp', space='XYZ', over=False, angle=(None,None), fig=0, figsize=None, dpi=None, dir="", fname="limits3D"):
    """ Plots the set of optimal colours
        (beware: mplot3d does not composite colours correctly, and cannot handle large sets)
    """
    if kind not in limits.keys() or space not in limits[kind].keys() or len(limits[kind][space]) == 0: return
    points = limits[kind][space]
    # figure
    fg = plt.figure(space)
    if not over:
        plt.close(fg)
        fg = plt.figure(space,figsize=figsize)
        ax = fg.add_subplot(111, projection='3d')
        ax.set_xlabel(space[index[space][0]])
        ax.set_ylabel(space[index[space][1]])
        ax.set_zlabel(space[index[space][2]])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(angle[0],angle[1])
        #ax.grid(False)
    else:
        ax = plt.gca()
    # plot
    print(len(points)," points")
    x = points[:,index[space][0]]
    y = points[:,index[space][1]]
    z = points[:,index[space][2]]
    colours = 'black' if over else np.clip(limits[kind]['RGB'],0,1)
    marker = 'x' if over else 'o'
    ax.scatter(x,y,z,color=colours,marker=marker,depthshade=False)
    # save
    if dir != "":
        fname = "%s/%s_%s.png"%(dir,fname,space)
        print("writing %s"%(fname))
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    if fig<0: plt.close(fg)

def plot_limits2D(kind='cmp', space='XYZ', dim=0, ranges={0:[None,None], 1:[None,None], 2:[None,None]}, over=False, fig=0, figsize=None, dpi=None, dir="", fname="limits2D"):
    """ Plots the set of optimal colours in a bin along a given dimension """
    if kind not in limits.keys() or space not in limits[kind].keys() or len(limits[kind][space]) == 0: return
    points = limits[kind][space]
    if ranges[dim][0] == None: ranges[dim][0] = points[:,dim].min()
    if ranges[dim][1] == None: ranges[dim][1] = points[:,dim].max()
    # figure
    fig_name = space+" "+space[dim]+"="+str(ranges[dim][0])+"-"+str(ranges[dim][1])
    fg = plt.figure(fig_name)
    if not over:
        plt.close(fg)
        fg = plt.figure(fig_name, figsize=figsize)
        ax = fg.add_subplot(111)
        ax.set_xlabel(space[(dim+1)%3])
        ax.set_ylabel(space[(dim+2)%3])
        ax.set_title("%s=%03i-%03i.png"%(space[dim],ranges[dim][0],ranges[dim][1]))
    else:
        ax = plt.gca()
    # plot
    indices = np.intersect1d(np.where(ranges[dim][0]<=points[:,dim]),np.where(points[:,dim]<=ranges[dim][1]))
    print(len(points[indices])," points")
    x = points[indices,(dim+1)%3]
    y = points[indices,(dim+2)%3]
    colours = 'black' if over else np.clip(limits[kind]['RGB'][indices,:],0,1)
    marker = 'x' if over else 'o'
    ax.scatter(x,y,color=colours,marker=marker)
    ax.set_xlim(ranges[(dim+1)%3])
    ax.set_ylim(ranges[(dim+2)%3])
    # save
    if dir != "":
        fname = "%s/%s_%s_%s=%03i-%03i.png"%(dir,fname,space,space[dim],ranges[dim][0],ranges[dim][1])
        print("writing %s"%(fname))
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    if fig<0: plt.close(fg)

def plot_limits2D_bins(space='XYZ', dim=1, binning=10, ranges={0:[-10,110], 1:[0,100], 2:[-10,110]}, figsize=None, dpi=None, dir=this_dir, fname='limits2D'):
    """ Plots the set of optimal colours in a set of bins along a given dimension """
    bins = np.linspace(ranges[dim][0],ranges[dim][1],int((ranges[dim][1]-ranges[dim][0])/(1.*binning)+1))
    for i in range(len(bins)-1):
        ranges_i = ranges
        ranges_i[dim] = [bins[i],bins[i+1]]
        plot_limits2D(kind='cmp',space=space,dim=dim,ranges=ranges_i,figsize=figsize,dpi=dpi)
        plot_limits2D(kind='ref',space=space,dim=dim,ranges=ranges_i,over=True,fig=-1,dir=dir,fname=fname)

def plot_Cmax(kind='cmp',figsize=None):
    """ Plots the surface of C(H,L) """
    fg = plt.figure('Cmax',figsize=figsize)
    ax = fg.add_subplot(111, projection='3d')
    ax.set_xlabel('H')
    ax.set_ylabel('L')
    ax.set_zlabel('C')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    X = limits[kind]['LCH'][:,2]
    Y = limits[kind]['LCH'][:,0]
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(np.shape(X))
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i,i] = limits[kind]['LCH'][i,1]
    ax.plot_surface(X, Y, Z)

#----------------
# use the limits
#----------------

def set_limits(l_step=1, l_min=360, l_max=780):
    """ Loads or computes the limits as needed (only needed once) """
    try:
        load_limits(l_step, l_min, l_max, dir=this_dir)
    except:
        print("couldn't load limits, computing them")
        find_limits(l_step, l_min, l_max, dir=this_dir)

triangulation = {'ref': {}, 'cmp': {}}

def triangulate(space,kind='cmp'):
    """ Computes the Delaunay triangulation of a set of points in 3D colour space
        (beware: scales poorly with data size)
    """
    global triangulation, limits
    triangulation[kind][space] = Delaunay(limits[kind][space])

def within_limits(point,space,kind='cmp'):
    """ Tests whether a point in space is within the limits
        (beware: scales poorly with data size)
    """
    global triangulation
    return triangulation[kind][space].find_simplex(point)>=0

def plot_triangulation(space,kind='cmp',figsize=None):
    """ Plots the triangulated mesh """
    global triangulation, limits
    points = limits[kind][space]
    tri = triangulation[kind][space]
    fig = plt.figure("hull_"+space+"_"+kind,figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel(space[index[space][0]])
    ax.set_ylabel(space[index[space][1]])
    ax.set_zlabel(space[index[space][2]])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=tri.simplices)


if hasattr(sys.modules[__name__],'colour'): get_limits_ref()
