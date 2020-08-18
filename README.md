This package provides some tools to work with colour in Python in a way that matches human perception. The adopted colour space is CIELCH, the cylindrical representation of CIELAB, with coordinates L = lightness in [0,100], C = chroma >= 0, H = hue angle in [0, 360].
The package relies on [NumPy](https://numpy.org) for array manipulations, [Matplotlib](https://matplotlib.org) for plotting, and optionally other colour packages listed below.

**convert** provides functions to convert a colour triplet between HCL and sRGB spaces, using either built-in [formulas](http://www.easyrgb.com/en/math.php) or wrappers to [colorspacious](https://pypi.python.org/pypi/colorspacious/) or [colour](http://colour-science.org) packages.

**gamut** provides functions to compute and visualize the boundary Cmax(L,H) of the sRGB gamut and of the full human gamut. These boundaries can be obtained by two methods: (1) for each point in the LCH plane, convert the colour to the target space (RGB or XYZ) and check whether it falls within the gamut; (2) get the contour points of the gamut in the space where it is naturally defined (RGB or XYZ), and convert them back to LCH space. For the latter method, to resample the function Cmax(L,H) on a regular grid the script relies on [scipy.interpolate](https://docs.scipy.org/doc/scipy/reference/interpolate.html). Both methods assume knowledge of the gamut in some space, this is trivial for the sRGB gamut, not for the human gamut hence the limits script below.  
The pre-computed gamuts at resolutions of 1 and 10 bins per unit L,H are included as Numpy binary files (30 MB).

**limits** provides functions to compute and visualize the limits of the human gamut (optimal colour stimuli, a.k.a. MacAdam limits). As explained on [Bruce Lindbloom's website](http://www.brucelindbloom.com/index.html?LabGamutDisplayHelp.html) they are obtained by considering all possible step-functions as spectral distributions. The conversion from light spectrum to XYZ colour is made with [colour](http://colour-science.org) (the computed limits can be compared with the crude triangulation provided by this package). To triangulate the limit surface in 3D the script relies on [scipy.spatial](https://docs.scipy.org/doc/scipy/reference/spatial.html) (this is required by method 1 above).  
The pre-computed limits at resolutions of 1, 5, 10, 20 nm are included as Numpy binary files (4.5 MB).

**slices** provides functions to draw slices in the colour space: L-H planes as a function of C (or for the maximum possible C), C-H planes as a function of L, and L-C planes as a function of H.

**maps** provides functions to generate colour maps for Matplotlib, and/or written as png images. Three kinds of cmaps are implemented, which separate the different perceptual dimensions: equi-luminant (varying only in H, at the max C), diverging (varying only in C between two hues H1 and H2), and mono-hue (varying only in L, at the maximum C). It includes utility functions for testing a cmap on mock data, and for displaying the path taken by a cmap in the colour space.

The package comes with five notebooks, which demonstrate the use of the five scripts.  
Static renderings of the notebooks can also be seen [on nbviewer](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/tree/master/):  
[1. Conversions between colour spaces](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/1.%20convert.ipynb)  
[2. Mapping the sRGB gamut](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/2.%20sRGB%20gamut.ipynb)  
[3. Mapping the human gamut](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/3.%20full%20gamut.ipynb)  
[4. Slicing the LCH space](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/4.%20slices.ipynb)  
[5. Making custom colour maps](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/5.%20cmaps.ipynb)  
Additional case study:
[5.1 Helix colour scheme](https://nbviewer.jupyter.org/github/gillesferrand/colourspace/blob/master/5.1%20cmaps%20helix.ipynb)

---
This package is provided "as is", with no warranty of any kind.  
The code can be freely used by whoever wants to explore modern colour theory.
