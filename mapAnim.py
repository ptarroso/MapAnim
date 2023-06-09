from osgeo import gdal
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from math import ceil


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
        raster = gdal.Open(filepath)
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
                # Apply the user-specified function to the input data
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


descr = ["Artificializado", "Culturas anuais de\noutono/inverno", 
         "Culturas anuais de\nprimavera/verão", "Outras áreas\nagrícolas", 
         "Sobreiro e\nAzinheira", "Eucalipto", "Outras folhosas", 
         "Pinheiro bravo", "Pinheiro manso", "Outras resinosas", "Matos", 
         "Vegetação\nherbácea\nespontânea", "Superfícies\nsem vegetação", 
         "Zonas húmidas", "Água"]

colors = ["#e31a1c", "#eba000", "#9effe4", "#f9f100", "#9f218a", "#2bf100",
          "#12c309", "#0c8006", "#174e11", "#1e636f", "#6f5e1a", "#d6cc90",
          "#787878", "#1973a7", "#362ecc"]

cls = [100, 211, 212, 213, 311, 312, 313, 321, 322, 323, 410, 420, 500, 610,
       620]


filepath = r"COSsim2021/COSsim_2021_N3_v0_TM06.tif"


#######################
#dp = MapCollection.fromrasterfile(filepath)

#Resize to fit image map
dt, p, nodata = resample_raster(filepath, 8)
dp = MapCollection(dt, dt[0][0])
dp.setColors(cls, colors)


# Add targets
targets = [[ 6500, 1000], [ 6500, 3000], [12500, 3000], 
           [ 6500, 6000], [12500, 6000], [14500, 6000], 
           [ 8500, 3000], [10500, 3000], [10500, 1000], 
           [14500, 1000], [ 8500, 6000], [10500, 6000],
           [ 8500, 1000], [12500, 1000], [14500, 3000]]





# set targets
inner_n = 5000
for i in range(len(cls)):
    cl = cls[i]
    print(cl)
    n = dp.getClass(cl).getSize()
    if n > inner_n:
        outer_n = ceil(n / inner_n)
        bs = spiral(outer_n, np.array(targets[i]), 55, 55)
        ts = [np.zeros(n), np.zeros(n)]
        start =  0
        j = 0
        for end in range(inner_n, n, inner_n):
            s = spiral(inner_n, [bs[0][j], bs[1][j]], 0.5, 0.5)
            ts[0][start:end] = s[0]
            ts[1][start:end] = s[1]
            j += 1
            start = end
        if end < n:
            s = spiral(n-end,  [bs[0][j], bs[1][j]], 0.5, 0.5)
            ts[0][end:n] = s[0]
            ts[1][end:n] = s[1]
    else:
        ts = spiral(n, np.array(targets[i]), 0.5, 0.5)
    dp.setTarget(cl, ts, targets[i])

# Because the orignal map was aggregated to a lower resolution, to mantain the
# original percentages I have to use the proportion occupied in each pixel

#totalpx = [dp.getClass(cl).size for cl in cls]
#percent = [x/sum(totalpx)*100 for x in totalpx]

ppx = [(p[cl][p[cl] != nodata]).sum() for cl in cls]
percent = [x/sum(ppx)*100 for x in ppx]

fbold = fm.FontProperties(fname="./Montserrat-SemiBold.ttf")
flight = fm.FontProperties(fname="./Montserrat-Light.ttf")

totalpx = [dp.getClass(cl).size for cl in cls]

plotlabels = [False for x in range(len(cls))]
showmap = None

figratio = 16/9 #155/80
width = 38.40 

outdir = "img"
flag = True
frame = 1

while flag:
    fname = f'{outdir}/{frame:06}.png'
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(width, width/figratio), dpi=100)
    
    # Title
    plt.text(7750, 7950, " ".join("Ocupação do Solo de"), alpha=1, 
             horizontalalignment='center', color="white", fontsize=50, 
             fontproperties=flight)
    plt.text(7750, 7450, " ".join("PORTUGAL CONTINENTAL"), alpha=1,
             horizontalalignment='center', color="white", fontsize=80, 
             fontproperties=fbold)
    lx = [7750-5800, 7750+5800]
    ly = [7400, 7400]
    plt.plot(lx, ly, linewidth=1, color="white")
    
    # Source and author
    plt.text(2750, 0, "Fonte: COSsim 2021\nAutor: Pedro Tarroso", alpha=1,
             color="white", fontsize=20, fontproperties=fbold)
    
    # Scale bar
    xs = 2750 #X coordinate scale
    ys = 350  #Y coordinate scale
    plt.plot([xs, xs+1250], [ys, ys], linewidth=2, color="white")
    plt.plot([xs, xs], [ys-50, ys+50], linewidth=2, color="white")
    plt.plot([xs+1250, xs+1250], [ys-50, ys+50], linewidth=2, color="white")
    plt.text(xs+1250/2, 375, "100 km", horizontalalignment='center',
             color="white", fontsize=20, fontproperties=fbold)
    
    # North Arrow
    xa = 2500 #X coordinate north arrow
    ya = 0    #Y coordinate north arrow
    sa = 300  #Size north arrow
    plt.fill([xa-sa*0.375, xa, xa], [ya, ya+sa, ya], color="white", linewidth=2)
    plt.plot([xa, xa+sa*0.375, xa], [ya, ya, ya+sa], color="white", linewidth=2)
    plt.text(xa, ya+sa*1.25, "N", horizontalalignment='center',
             color="white", fontsize=20, fontproperties=fbold)
    
    # Plot class labels with alpha related to pixels on target
    for i in range(len(cls)):
        alpha = plotlabels[i] / totalpx[i]
        if percent[i] > 0.1:
            lbl = "{0}\n({1:.1f}%)".format(descr[i], percent[i])
        else:
            val = 1/10**lead_zeros(percent[i])
            lbl = "{0}\n(<{1}%)".format(descr[i], val)
        x,y = targets[i]
        plt.text(x, y-1350, lbl, alpha=alpha, horizontalalignment='center',
                 color="white", fontsize=24, fontproperties=fbold)
    
    for i in range(len(cls)):
        dt = dp.getClass(cls[i])
        x,y = dt.getCoords()
        col = dt.getColor()
        plt.scatter(x, y, s=0.005, c = col, zorder=100)
    
    # Replot original map when transition is over
    if sum(plotlabels) / sum(totalpx) > 0.9995:
        if showmap is None:
            showmap = frame
        else:
            alpha = (frame-showmap) / 150
            if alpha > 1:
                alpha = 1
            for cl in dp.getClasses():
                dt = dp.getClass(cl)
                x,y = dt.getOriginalCoords()
                col = dt.getColor()
                plt.scatter(x, y, s=0.005, c = col, alpha=alpha)
    
    axes=plt.gca()
    axes.set_xlim([0, 15500])
    axes.set_ylim([0, 8000])
    axes.set_aspect("equal")
    axes.axis("off")
    plt.tight_layout()
    plt.savefig(fname) #, transparent=True)
    plt.close(fig)
    frame += 1
    
    
    # Start moving after frame 50
    mag = 1
    if frame < 300:
        mag = (frame-50)/250
    if frame > 50:
        for i in range(len(cls)):
            plotlabels[i] = dp.getClass(cls[i]).move(mag, tol=0.001)


