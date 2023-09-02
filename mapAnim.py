#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Raster Map Animation

This script serves as a data visualization tool for categorical raster maps.
It animates pixels with data into tightly packed spirals, providing a visually
appealing representation of the proportions of each class. The script allows
users to fine-tune various parameters and incorporate map elements and labels
through user-defined text files.

Author: Pedro Tarroso
Date: August 30, 2023

License: GNU General Public License (GPL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import json
from osgeo import gdal
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from math import ceil
import argparse

# To avoid FutureWarning
gdal.DontUseExceptions()

class MapClass():
    """
    The MapClass class is responsible for storing the coordinates of a single 
    class, such as pixel values, and facilitating linear movement of each 
    point from its original position to a target position. It holds important 
    information, such as the class color and the original, current, and target
    positions for each point.
    """
    def __init__(self, classname, x, y, color):
        """
        classname: A string representing the name of the class
        x: A vector of x-coordinate positions
        y: A vector of y-coordinate positions (with the same length as X)
        color: The color code for the class.
        """
        if len(x) != len(y):
            raise ValueError("x and y must be of the same size")
        if type(x) != type(np.zeros(0)) or type(y) != type(np.zeros(0)):
            try:
                x = np.array(x)
                y = np.array(y)
            except:
                print("x and y must be numpy arrays")
        self.name = classname
        self.x = x
        self.y = y
        self.ox = x
        self.oy = y
        self.size = len(self.x)
        self.color = color
        self.dist = np.zeros(len(x))
        self.tx = np.zeros(len(x))
        self.ty = np.zeros(len(x))
    
    def getSize(self):
        """ Returns the size of the class (number of points)"""
        return self.size
    
    def getColor(self):
        """ Returns the current color of the class. """
        return self.color
    
    def getCoords(self):
        """ Returns a list of the [[x],[y]] current positions."""
        return [self.x, self.y]
    
    def getOriginalCoords(self):
        """ Returns a list of the [[x],[y]] original positions."""
        return [self.ox, self.oy]
    
    def getTargetCoords(self):
        """ Returns a list of the [[x],[y]] target positions."""
        return [self.tx, self.ty]
    
    def getDist(self):
        """ Returns the Euclidean distance from current postion to target."""
        return(self.dist**0.5)
    
    def setTargets(self, x, y, t_crd=None):
        """
        The setTargets() method is responsible for setting the target positions
        for each point in the class and reordering the data in relation to 
        their distance. If the optional argument 't_crd' is given (a list with 
        coordinate pairs), then the reordering is calculated based on the 
        distance to that point. By default, each point is reordered based on 
        the distance to its own individual target.
        
        The method takes the following parameters:
        x: A list of X target positions for each point.
        y: A list of Y target positions for each point.
        t_crd (optional): A list of coordinate pairs that sets the order based
            on distance to the current position.

        After this method is called, the class data is updated with the new 
        target positions and the points are reordered based on the distance
        to the specified point or their own target.
        """
        if len(x)==len(self.x) & len(y)==len(self.y):
            try:
                self.tx = np.array(x)
                self.ty = np.array(y)
            except:
                print("x and y must be numpy arrays or coercible to numpy array.")
        else:
            raise ValueError("x and y must be of the same size of original coordinates")
        self.reorder(t_crd)
    
    def calcDist(self):
        """
        The calcDist() method is responsible for calculating the current 
        distance to the targets. This method is primarily used for reordering
        the data and is designed to avoid calculating the square root of the
        distance for speed optimization.
        """
        d = (self.x - self.tx)**2 + (self.y - self.ty)**2
        self.dist = d
    
    def setColor(self, color):
        """ Sets the current color of the class. """
        self.color = color
    
    def reorder(self, xy = None):
        """
        The reorder() method is responsible for reordering all points in the 
        class in respect to the distance to the targets. This method is 
        primarily used internally, but may be useful in situations where the
        behavior of each point (e.g. display color) depends on its position in
        relation to the targets or a specific point.

        xy (optional): A coordinate pair that sets the order based on distance
            to the single point. If xy is None, all points are reordered in 
            respect to their individual targets.
        
        The method reorders the points by calculating the distance to the 
        targets (or the single point specified by xy) and reordering the 
        points based on this distance. This ensures that the points are 
        arranged in the correct order for any subsequent operations or 
        analyses that depend on the spatial relationships between the points.
        """
        if xy is None:
            self.calcDist()
            d = self.dist
        elif len(xy) == 2:
            d = (self.x - xy[0])**2 + (self.y - xy[1])**2
        order = d.argsort()
        self.x = self.x[order]
        self.y = self.y[order]
        self.tx = self.tx[order]
        self.ty = self.ty[order]
        self.ox = self.ox[order]
        self.oy = self.oy[order]
        self.calcDist()
    
    def move(self, mag=0.1, tol=0.0001):
        """ 
        The move() method is responsible for moving each point towards its 
        target by a user-defined magnitude. This method is used to animate the
        movement of the points in the class towards their targets.
        
        The method takes the following parameters:

        mag: The magnitude of the movement, which determines how far each 
             point moves towards its target in each iteration.
        tol: The tolerance used to determine if the point has reached its 
             target position. If the distance between the point and its target
             is less than or equal to the tolerance, the point is considered 
             to have reached its target position.
        
        The method updates the position of each point in the class, moving it 
        towards its target by the specified magnitude. It then checks if the 
        point has reached its target position by calculating the distance 
        between the point and its target, and comparing it to the specified 
        tolerance. If the distance is less than or equal to the tolerance, the
        point is considered to have reached its target position and the method
        stops moving that point.
        Returns the number of non-moving pixels.
        """
        s = self.size
        self.calcDist()
        x = self.x
        y = self.y
        
        m = self.dist > tol
        ms = m.sum()
        if ms != 0:
            mag = (1-1/(1+np.exp(-0.0001*(np.where(m)[0]-(s-ms+10000)))))*mag
            vx = np.zeros(s)
            vx[m] = (self.tx[m] - x[m]) * mag
            vy = np.zeros(self.size)
            vy[m] = (self.ty[m] - y[m]) * mag
            
            self.x = x + vx
            self.y = y + vy
        return s-ms

class MapCollection():
    """
    The MapCollection class is responsible for managing a collection of 
    MapClass objects. It maintains and updates a list of classes (pixel
    values) and provides methods for working with this list of classes.

    The class is initialized with a numpy array (raster matrix) from which
    unique classes are extracted. Alternatively, it can be initialized from a
    raster file. The unique classes extracted from the input data are 
    represented as MapClass objects and stored as a list in the MapCollection
    object.

    This class is useful for managing collections of MapClass objects and 
    provides a convenient interface for working with multiple classes at once.

    Note that the MapCollection class works with pixel positions in the input
    array (rows and columns) and does not preserve the real-world coordinate
    values of the data.
    """
    def __init__(self, arr, nodata=0):
        """
        The default initialization of a MapCollection object requires a numpy 
        array arr as input. This array represents the raster data and should 
        contain unique class values to be extracted. The array is typically 
        2D, and it will be flattened to create a 1D array of unique values.
        
        The MapCollection class will extract the unique values from the input 
        array and create a MapClass object for each unique value.
        """
        nrow, ncol = arr.shape
        arr = arr.flatten()
        msk = arr != nodata
        z = arr[msk]
        x = np.tile(range(ncol), nrow)[msk]
        y = np.repeat(range(nrow-1, -1, -1), ncol)[msk]
        cl, count = np.unique(z, return_counts=True)
        data = []
        for i in range(len(cl)):
            cls = cl[i]
            mc = MapClass(cls, x[z==cls], y[z==cls], "#000000")
            data.append(mc)
        self.data = data
    
    @classmethod
    def fromrasterfile(cls, filename, nodata=0, rtype='int16'):
        """
        The fromrasterfile() method of the MapCollection class provides a 
        practical way to initialize a MapCollection object directly from a 
        raster file.
        
        filename: path to filename
        nodata: nodata value to be used (defaults to 0)
        rtype: data type of the array to be extracted from raster
        """
        raster = gdal.Open(filename)
        rst = raster.ReadAsArray().astype(rtype)
        return cls(rst, nodata)
    
    def getClasses(self):
        """ Get the names of the unique classes stored."""
        return [x.name for x in self.data]
    
    def getClass(self, cl):
        """ Returns the MapClass object of the 'cl' class."""
        cl = [x for x in self.data if x.name == cl]
        if len(cl) == 1:
           return cl[0]
        else:
            raise ValueError("Class " + str(cl) + " not found.")
     
    def calcDists(self):
        """ Calculate current distances to targets for all points in all classes."""
        [x.calcDist() for x in self.data]
        
    def reorder(self):
        """ Reorder the data in all classes. """
        [x.reorder() for x in self.data]
    
    def setTarget(self, cl, target, t_crd=None):
        """
        The getTarget() method is used to set targets for a specific class. It
        takes two required arguments: cl, which is the identifier for the 
        class for which the targets are being set, and target, which is a 2D 
        array with the x-coordinates in the first element and y-coordinates in
        the second element.
        
        Optionally, you can also provide a t_crd argument. If this argument is 
        None (which is the default), the method will calculate the distance 
        from each point to each target in the order given. If t_crd is a 
        coordinate pair in the form [x, y], then the distance will be 
        calculated from each point to this single center point.
        """
        dt = self.getClass(cl)
        s = dt.getSize()
        if len(target) == 2:
            if len(target[0]) == s & len(target[1]) == s:
                dt.setTargets(target[0], target[1], t_crd)
    
    def setColors(self, cls, colors):
        """
        The setColors() method is used to set the color of each class. It
        takes 2 arguments:
        cls: a list of classes to set colors
        colors: a list of corresponding color to each class

        This method can be called during animation to change the color of the 
        class.
        """

        if len(cls) != len(colors):
            raise ValueError("Classes 'cls' and colors must have same length")
        for i in range(len(cls)):
            dt = self.getClass(cls[i])
            dt.setColor(colors[i])

def lead_zeros(n, max=4):
    """
    Simple utility function that takes a number as input and returns the 
    number of zeros between the decimal separator and the first non-zero 
    digit.
    """
    s = '{0:.{1}f}'.format(round(n, max), max).split('.')
    if int(s[0]) != n:    
        return len(s[1]) - len(s[1].lstrip('0'))
    return(0)

def spiral(n, origin = [0.0, 0.0], arc=0.1, sep=0.1):
    """
    This function is used to distribute a given number of 'n' points along a 
    spiral curve, starting at a given 'origin' point and following a curvature
    'arc' with a specified separation distance sep between points.

    The function takes three arguments:
    origin: a list or tuple of two coordinates defining the starting point of 
            the spiral;
    arc: determines the curvature of the spiral and is given as a float;
    sep: is the separation distance between points and is also given as a 
         float.

    The function returns a list of two sub-lists [[x], [y]] containing the 
    coordinates of the points in the spiral curve. The length of each sub-list
    is equal to the number of points n in the spiral.
    """
    r = arc
    b = sep / (2* np.pi)
    phi = r / b
    pnt = [np.repeat(origin[0], n), np.repeat(origin[1], n)]
    for i in range(1, n):
        pnt[0][i] = origin[0]+r*np.cos(phi)
        pnt[1][i] = origin[1]+r*np.sin(phi)
        phi = phi + arc/r
        r = b * phi
    return pnt

def resample_raster(raster_path, factor):
    """
    Resamples a raster by a desired factor and returns a 2D array of values, a
    dictionary of arrays for the proportions of each class in relation to the
    original raster size, and the nodata value used.

    Parameters:
    raster_path: Path and filename of the raster to be read.
    factor: The downscaling factor.
    """
    # Open the raster dataset
    raster = gdal.Open(raster_path)
    
    # Get the input raster dimensions
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    
    # Calculate the output raster dimensions
    out_rows = int(rows / factor)
    out_cols = int(cols / factor)
    out_px = out_rows * out_cols
    
    # Get NoData value
    nodata = raster.GetRasterBand(1).GetNoDataValue()
    
    arr = np.array([nodata] * out_px).reshape(out_rows, out_cols)
    p = {}
    # Loop over the output raster and fill in the values
    for i in range(out_rows):
        print("Resampling: %3.1f%%" % (i/out_rows*100), end = "\r")
        for j in range(out_cols):
            #k = j + (i*out_cols)
            # Get the input pixel values for the resampled region
            input_data = raster.ReadAsArray(j * factor, i * factor, factor, factor)
            # Remove no data
            input_data = input_data[input_data != nodata]
            if len(input_data) > 0:
                # Apply the summarising function to the input data
                values, counts = np.unique(input_data, return_counts=True)
                ind = np.argmax(counts)
                arr[i][j] = values[ind]
                for vi in range(len(values)):
                    val = values[vi]
                    if val not in p.keys():
                        p[val] = np.array([nodata] * out_px).reshape(out_rows, out_cols)
                    p[val][i][j] = counts[vi]/len(input_data)
    print()
    return([arr, p, nodata])

class DataDescriptor:
    """
    The DataDescriptor class is designed to manage descriptions, color codes, 
    and targets associated with each class value. This class offers methods 
    for adding new classes and retrieving essential information from each data
    class. Additionally, it includes a method to simplify the process of 
    extracting class descriptions from a CSV file."
    """

    def __init__(self):
        self.cls = []
        self.descr = []
        self.colors = []
        self.targets = []
    
    def addClass(self, cls, descr, color, target):
        """
        Adds a class with relvant and mandatory associated data.
        """
        if cls not in self.cls:
            self.cls.append(int(cls))
            self.descr.append(descr.replace('\\n', '\n'))
            self.colors.append(color)
            self.targets.append([int(x) for x in target])
        else:
            raise ValueError(f"Can't add {cls} because it is already available.")

    @classmethod
    def fromfile(cls, filename):
        """
        This class method enables the creation of a DataDescriptor object from
        a CSV file. The CSV file should be comma-separated and contain the 
        following fields: 
            [value, description, color, target_x, target_y]
        including a header row. While the order of the fields is flexible, it's
        important to use these exact field names for successful data extraction.
        """
        header = True
        order = ["value", "description", "color", "target_x", "target_y"]
        dd = cls()
        with open(filename, "r") as stream:
            for line in stream:
                line = line.strip().split(",")
                if header:
                    try:
                        line = [x.lower() for x in line]
                        i = [line.index(x) for x in order]
                        header = False
                    except ValueError as err:
                        print(err)
                else:
                    dd.addClass(line[i[0]], line[1], line[2], line[3:5])
        return dd
    
    def getSize(self):
        """ Get the number of available classes."""
        return len(self.cls)

    def classExists(self, cls):
        """ Test if a particular class exists."""
        if cls in self.cls:
            return True
        else:
            raise ValueError(f"Class {cls} not available.")

    def getClass_by_index(self, i):
        """ Get the class name by its index."""
        if i < self.getSize():
            return self.cls[i]

    def modifyClass(self, cls, descr=None, color=None, target=None):
        """ Method for modifying an existing class."""
        if self.classExists(cls):
            i = self.cls.index(cls)
            if descr:
                self.descr[i] = descr
            if color:
                self.color[i] = color
            if target:
                self.targets[i] = target

    def rmClass(self, cls):
        """ Method for removing an existing class."""
        if self.classExists(cls):
            i = self.cls.index(cls)
            self.cls.pop(i)
            self.descr.pop(i)
            self.color.pop(i)
            self.targets.pop(i)

    def getDescriptor(self, cls):
        """ Get the description of an existing class. """
        if self.classExists(cls):
            i = self.cls.index(cls)
            return(self.descr[i])

    def getColor(self, cls):
        """ Get the color of an existing class. """
        if self.classExists(cls):
            i = self.cls.index(cls)
            return(self.colors[i])
    
    def getTarget(self, cls):
        """ Get the target of an existing class. """
        if self.classExists(cls):
            i = self.cls.index(cls)
            return(self.targets[i])

class ElementsPlot():
    """
    The ElementsPlot class serves as a container for storing additional 
    elements intended for display on a map. These elements may include text,
    lines, as well as features such as scale bars and north arrow.
    It provides a class method for easy build from JSON file.
    """
    def __init__(self):
        self.elems = []

    def addText(self, x, y, label, params={}):
        """ 
        Adds a text element at (x,y). 
        'Params' is a dict with relevant plotting parameters for text 
        (e.g. fontsize).
        """
        self.elems.append({"type": "text",
                           "x": x,
                           "y": y,
                           "label": label,
                           "params": params
                           })

    def addLine(self, x, y, params={}):
        """ 
        Adds a line element at (x,y). 
        'Params' is a dict with relevant plotting parameters for lines
        (e.g. linewidth).
        """
        self.elems.append({"type": "line",
                           "x": x,
                           "y": y,
                           "params": params
                           })
    
    def addScalebar(self, label, x, y, xsize, ysize, params={}):
        """ 
        Adds a scale bar element at (x,y) with user defined size. 
        'Params' is a dict with relevant plotting parameters for scalebar
        (e.g. linewidth, fontsize).
        """
        self.elems.append({"type": "scalebar",
                           "label": label,
                           "x": x,
                           "y": y,
                           "xsize": xsize,
                           "ysize": ysize,
                           "params": params
                           })
        
    def addNortharrow(self, x, y, size, params={}):
        """
        Adds a north arrow element at (x,y) with user defined size. 
        'Params' is a dict with relevant plotting parameters for northarrow
        (e.g. linewidth, fontsize).
        """
        self.elems.append({"type": "northarrow",
                           "x": x,
                           "y": y,
                           "size": size,
                           "params": params
                           })

    def addClasslabel(self, x, y, label, params={}):
        """
        Adds a class label ellement at (x,y). 
        'Params' is a dict with relevant plotting parameters for text
        (e.g. fontsize).
        """
        self.elems.append({"type": "classlabel",
                           "x": x,
                           "y": y,
                           "label": label,
                           "params": params
                           })
    @classmethod
    def fromJSON(cls, filename):
        """
        This class method facilitates the creation of an ElementsPlot object 
        from a well-structured JSON file. The JSON file should adhere to a 
        specific format that includes relevant properties. Please refer to the
        provided example for the correct format and to identify commonly used 
        elements.
        """
        with open(filename, "r") as stream:
            elems = json.load(stream)
        for e in elems:
            if e["params"]:
                if "fontproperties" in e["params"]:
                    font = e["params"]["fontproperties"]
                    e["params"]["fontproperties"] = fm.FontProperties(fname=font)
        e = cls()
        e.elems = elems
        return e
    
    def plot_elements(self):
        """ Plots all elements with the exception of 'classlabels'."""
        for e in self.elems:
            if e["type"] == "text":
                plt.text(e["x"], e["y"], e["label"], **e["params"])
            elif e["type"] == "line":
                plt.plot(e["x"], e["y"], **e["params"])
            elif e["type"] == "scalebar":
                x = e["x"]
                y = e["y"]
                xs = e["xsize"]
                ys = e["ysize"]
                plt_params = {key: value for key, value in e["params"].items() if key in ["color", "linewidth"]}
                plt.plot([x, x+xs], [y, y], **plt_params)
                plt.plot([x, x], [y-ys, y+ys], **plt_params)
                plt.plot([x+xs, x+xs], [y-ys, y+ys], **plt_params)
                txt_params = {key: value for key, value in e["params"].items() if key in ["color", "fontsize", "fontproperties"]}
                plt.text(x+xs/2, y*1.1, e["label"], horizontalalignment="center", **txt_params)
            elif e["type"] == "northarrow":
                xa = e["x"]
                ya = e["y"]
                sa = e["size"]
                fill_params = {key: value for key, value in e["params"].items() if key in ["color"]}
                plt.fill([xa-sa*0.375, xa, xa], [ya, ya+sa, ya], **fill_params)
                plt_params = {key: value for key, value in e["params"].items() if key in ["color", "linewidth"]}
                plt.plot([xa, xa+sa*0.375, xa], [ya, ya, ya+sa], **plt_params)
                txt_params = {key: value for key, value in e["params"].items() if key in ["color", "fontsize", "fontproperties"]}
                plt.text(xa, ya+sa*1.25, "N", horizontalalignment="center", **txt_params)

    def plot_classlabels(self, lbl, pos = [0,0], alpha=0):
        """
        Plots all available class labels. It allows to control transparency
        of each label for animation purposes.
        """
        for e in self.elems:
            if e["type"] == "classlabel":
                x = pos[0] + e["x"]
                y = pos[1] + e["y"]
                plt.text(x, y, lbl, alpha=alpha, **e["params"])


def main(raster_file, arc, iarc, inner_n, width, height, xlim, ylim, elem_file,
         class_file, burnin, burnout, fadein, magfactor, accel, outdir, 
         resample=None):
    """ The main function that reads data, animates and exports each frame."""

    # Read the elements
    elems = ElementsPlot.fromJSON(elem_file)

    # Read the parameters
    dd = DataDescriptor.fromfile(class_file)

    # Read the raster
    if resample:
        dt, p, nodata = resample_raster(raster_file, resample)
        dp = MapCollection(dt, dt[0][0])
        ppx = [(p[cl][p[cl] != nodata]).sum() for cl in dd.cls]
        percent = [x/sum(ppx)*100 for x in ppx]
        totalpx = [dp.getClass(cl).size for cl in dd.cls]
    else:
        dp = MapCollection.fromrasterfile(raster_file)
        totalpx = [dp.getClass(cl).size for cl in dd.cls]
        percent = [x/sum(totalpx)*100 for x in totalpx]

    dp.setColors(dd.cls, dd.colors)

    # Set the targets in the spiral for each pixel
    for i in range(dd.getSize()):
        print("Setting targets for each pixel/class: %3.1f%%" % ((i+1)/dd.getSize()*100), end = "\r")
        cl = dd.getClass_by_index(i)
        n = dp.getClass(cl).getSize()
        target = dd.getTarget(cl)
        if n > inner_n:
            outer_n = ceil(n / inner_n)
            bs = spiral(outer_n, np.array(target), arc, arc)
            ts = [np.zeros(n), np.zeros(n)]
            start =  0
            j = 0
            for end in range(inner_n, n, inner_n):
                s = spiral(inner_n, [bs[0][j], bs[1][j]], iarc, iarc)
                ts[0][start:end] = s[0]
                ts[1][start:end] = s[1]
                j += 1
                start = end
            if end < n:
                s = spiral(n-end,  [bs[0][j], bs[1][j]], iarc, iarc)
                ts[0][end:n] = s[0]
                ts[1][end:n] = s[1]
        else:
            ts = spiral(n, np.array(target), iarc, iarc)
        dp.setTarget(cl, ts, target)
    print()

    plotlabels = [False for x in range(dd.getSize())]
    showmap = None
    flag = True
    frame = 1
    amode = "Burn-in"
    # Animate
    while flag:
        fname = f'{outdir}/{frame:06}.png'
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        
        elems.plot_elements()
        
        # Plot class labels with alpha related to pixels on target
        for i in range(dd.getSize()):
            cl = dd.getClass_by_index(i)
            alpha = plotlabels[i] / totalpx[i]
            if percent[i] > 0.1:
                lbl = "{0}\n({1:.1f}%)".format(dd.getDescriptor(cl), percent[i])
            else:
                val = 1/10**lead_zeros(percent[i])
                lbl = "{0}\n(<{1}%)".format(dd.getDescriptor(cl), val)
            elems.plot_classlabels(lbl, dd.getTarget(cl), alpha=alpha)
        
        for i in range(dd.getSize()):
            dt = dp.getClass(dd.cls[i])
            x,y = dt.getCoords()
            col = dt.getColor()
            plt.scatter(x, y, s=0.005, c = col, zorder=100)
        
        # Replot original map when transition is over
        active_pixels = 1 - sum(plotlabels) / sum(totalpx)
        if  active_pixels < 0.0005:
            amode = "Finishing"
            if showmap is None:
                showmap = frame
            else:
                alpha = (frame-showmap) / fadein
                if alpha > 1:
                    alpha = 1
                for cl in dp.getClasses():
                    dt = dp.getClass(cl)
                    x,y = dt.getOriginalCoords()
                    col = dt.getColor()
                    plt.scatter(x, y, s=0.005, c = col, alpha=alpha)
                if frame > showmap + burnout + fadein:
                    flag = False
        
        axes=plt.gca()
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_aspect("equal")
        axes.axis("off")
        plt.tight_layout()
        plt.savefig(fname) #, transparent=True)
        plt.close(fig)

        print(f"{amode} | Frame: {frame} | Active pixels: {active_pixels*100:.2f}%", end = "\r")

        frame += 1
        
        # Start moving after burnin
        if frame < (accel + burnin):
            mag = ((frame-burnin)/accel) * magfactor
        if frame > burnin:
            amode = "Animating"
            for i in range(dd.getSize()):
                cl = dd.getClass_by_index(i)
                plotlabels[i] = dp.getClass(cl).move(mag, tol=0.001)
    print()

# Comand line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("raster", type=str,
                    help="The input raster map with integer classes (TIF format).")
parser.add_argument("classdata", type=str,
                    help="The class information for each of the classes in the raster (CSV format).")
parser.add_argument("elements", type=str,
                    help="The extra elements to plot (json format).")
parser.add_argument("outdir", type=str,
                    help="The output directory to save each frame.")
parser.add_argument("-s", "--size", nargs=2, type=int, default= [1920, 1080],
                    help="The output figure given as 'width height'.")
parser.add_argument("-x", "--xlim", nargs=2, type=int, default=[0, 15500],
                    help="The limits for X axis in pixels. Note that position of elements are using this axis.")
parser.add_argument("-y", "--ylim", nargs=2, type=int, default=[0, 8000],
                    help="The limits for Y axis in pixels. Note that position of elements are using this axis.")
parser.add_argument("-bi", "--burnin", type=int, default=50,
                    help="Number of frames without movement in the begining of the animation.")
parser.add_argument("-bo", "--burnout", type=int, default=50,
                    help="Number of frames without movement at the end of the animation.")
parser.add_argument("-f", "--fadein", type=int, default=250,
                    help="Number of frames for fade in the original raster in the map.")
parser.add_argument("-m", "--mfactor", type=float, default=1.0,
                    help="Magnitude factor for the movement of each pixel (higher value implies faster movement)")
parser.add_argument("-a", "--accel", type=int, default=250,
                    help="Number of frames for movement accelaration at the begining of the animation.")
parser.add_argument("-i", "--innern", type=int, default=5000,
                    help="Number of points of the inner spiral (each dot of the main spiral).")
parser.add_argument("-r", "--resample", type=int,
                    help="Sets the resampling factor for downscaling the raster.")
parser.add_argument("-oa", "--arc", type=float, default=55,
                    help="Controls the outer spiral arc and separation.")
parser.add_argument("-ia", "--iarc", type=float, default=0.5,
                    help="Controls the inner spiral arc and separation.")
args = parser.parse_args()


if __name__ == "__main__":
    width, height = args.size
    main(args.raster, args.arc, args.iarc, args.innern, width, height, 
         args.xlim, args.ylim, args.elements, args.classdata, args.burnin, 
         args.burnout, args.fadein, args.mfactor, args.accel, args.outdir, 
         args.resample)
