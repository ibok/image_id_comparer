#!/usr/bin/env python

""" Description: An interactive program where the user may input a number
corresponding to an image type which is compared to a program's
classification. """

def type_of(x): #Returns string version of type
    return {
        1 : 'spot',
        2 : 'other',
        3 : 'worm',
        4 : 'track',
    }[x]

def output(str):
    return x.write(str + '\n')

class Blob:
    """Class that defines a 'blob' in an image: the contour of a set of pixels
       with values above a given threshold."""

    def __init__(self, x, y):
        """Define a counter by its contour lines (an list of points in the xy
           plane), the contour centroid, and its enclosed area."""
        self.x = x
        self.y = y
        self.xc = np.mean(x)
        self.yc = np.mean(y)

        # Find the area inside the contour
        self.area = 0.
        n = len(x)
        for i in range(0, n):
            self.area += 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])

    def distance(self, blob):
        """Calculate the distance between the centroid of this blob contour and
           another one in the xy plane."""
        return np.sqrt((self.xc - blob.xc)**2 + (self.yc-blob.yc)**2)

class BlobGroup:
    """A list of blobs that is grouped or associated in some way, i.e., if
       their contour centroids are relatively close together."""

    def __init__(self):
        """Initialize a list of stored blobs and the bounding rectangle which
        defines the group."""
        self.blobs = []
        self.xmin =  1e10
        self.xmax = -1e10
        self.ymin =  1e10
        self.ymax = -1e10

    def addBlob(self, blob):
        """Add a blob to the group and enlarge the bounding rectangle of the
           group."""
        self.blobs.append(blob)
        self.xmin = min(self.xmin, blob.x.min())
        self.xmax = max(self.xmax, blob.x.max())
        self.ymin = min(self.ymin, blob.y.min())
        self.ymax = max(self.ymax, blob.y.max())
        self.cov  = None

    def getBoundingBox(self):
        """Get the bounding rectangle of the group."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def getSquareBoundingBox(self):
        """Get the bounding rectangle, redefined to give it a square aspect
           ratio."""
        xmin, xmax, ymin, ymax = (self.xmin, self.xmax, self.ymin, self.ymax)
        xL = np.abs(xmax - xmin)
        yL = np.abs(ymax - ymin)
        if xL > yL:
            ymin -= 0.5*(xL-yL)
            ymax += 0.5*(xL-yL)
        else:
            xmin -= 0.5*(yL-xL)
            xmax += 0.5*(yL-xL)
        return (xmin, xmax, ymin, ymax)

    def getSubImage(self, image):
        """Given an image, extract the section of the image corresponding to
           the bounding box of the blob group."""
        ny,nx = image.shape
        x0,x1,y0,y1 = self.getBoundingBox()

        # Account for all the weird row/column magic in the image table...
        i0,i1 = [ny - int(t) for t in (y1,y0)]
        j0,j1 = [int(t) for t in (x0,x1)]

        # Add a pixel buffer around the bounds, and check the ranges
        buf = 1
        i0 = 0 if i0-buf < 0 else i0-buf
        i1 = ny-1 if i1 > ny-1 else i1+buf
        j0 = 0 if j0-buf < 0 else j0-buf
        j1 = nx-1 if j1 > nx-1 else j1+buf

        return image[i0:i1, j0:j1]

    def getRawMoment(self, image, p, q):
        """Calculate the image moment given by
           M_{ij}=\sum_x\sum_y x^p y^q I(x,y)
           where I(x,y) is the image intensity at location x,y."""
        nx,ny = image.shape
        Mpq = 0.
        if p == 0 and q == 0:
            Mpq = np.sum(image)
        else:
            for i in range(0,nx):
                x = 0.5 + i
                for j in range(0,ny):
                    y = 0.5 + j
                    Mpq += x**p * y**q * image[i,j]
        return Mpq

    def getCovariance(self, image):
        """Get the raw moments of the image region inside the bounding box
           defined by this blob group and calculate the image covariance
           matrix."""
        if self.cov is None:
            subImage = self.getSubImage(image).transpose()
            M00 = self.getRawMoment(subImage, 0, 0)
            M10 = self.getRawMoment(subImage, 1, 0)
            M01 = self.getRawMoment(subImage, 0, 1)
            M11 = self.getRawMoment(subImage, 1, 1)
            M20 = self.getRawMoment(subImage, 2, 0)
            M02 = self.getRawMoment(subImage, 0, 2)
            xbar = M10/M00
            ybar = M01/M00
            self.cov = np.vstack([[M20/M00 - xbar*xbar, M11/M00 - xbar*ybar],
                                  [M11/M00 - xbar*ybar, M02/M00 - ybar*ybar]])
        return self.cov

    def getPrincipalMoments(self, image):
        """Return the maximum and minimum eigenvalues of the covariance matrix,
           as well as the angle theta between the maximum eigenvector and the
           x-axis."""
        cov = self.getCovariance(image)
        u20 = cov[0,0]
        u11 = cov[0,1]
        u02 = cov[1,1]

        theta = 0.5 * np.arctan2(2*u11, u20-u02)
        l1 = 0.5*(u20+u02) + 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        l2 = 0.5*(u20+u02) - 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        return l1, l2, theta

def findBlobs(image, threshold, minArea=2.):
    """Pass through an image and find a set of blobs/contours above a set
       threshold value.  The minArea parameter is used to exclude blobs with an
       area below this value."""
    blobs = []
    ny, nx = image.shape

    # Find contours using the Marching Squares algorithm in the scikit package.
    contours = measure.find_contours(image, threshold)
    for contour in contours:
        x = contour[:,1]
        y = ny - contour[:,0]
        blob = Blob(x, y)
        if blob.area >= minArea:
            blobs.append(blob)
    return blobs

def groupBlobs(blobs, maxDist):
    """Given a list of blobs, group them by distance between the centroids of
       any two blobs.  If the centroids are more distant than maxDist, create a
       new blob group."""
    n = len(blobs)
    groups = []
    if n >= 1:
        # Single-pass clustering algorithm: make the first blob the nucleus of
        # a blob group.  Then loop through each blob and add either add it to
        # this group (depending on the distance measure) or make it the
        # nucleus of a new blob group
        bg = BlobGroup()
        bg.addBlob(blobs[0])
        groups.append(bg)

        for i in range(1, n):
            bi = blobs[i]
            isGrouped = False
            for group in groups:
                # Calculate distance measure for a blob and a blob group:
                # blob just has to be < maxDist from any other blob in the group
                for bj in group.blobs:
                    if bi.distance(bj) < maxDist:
                        group.addBlob(bi)
                        isGrouped = True
                        break
            if not isGrouped:
                bg = BlobGroup()
                bg.addBlob(bi)
                groups.append(bg)

    return groups

""" Figure that graphical display to show image with event """
def showFigure(path):
    contours = 40
    # Load the image and convert pixel values to grayscale intensities
    filename = path
    img = Image.open(filename).convert("L")
    image = []
    pix = img.load()

    # Stuff image values into a 2D table called "image"
    nx = img.size[0]
    ny = img.size[1]
    x0, y0, x1, y1 = (0, 0, nx, ny)
    for y in xrange(ny):
        #drawProgressBar(float(y+1)/ny)
        image.append([pix[x, y] for x in xrange(nx)])
    sys.stdout.write("\n")

    image = np.array(image, dtype=float)

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    #contours = measure.find_contours(image, args.contours)
    blobs = findBlobs(image, contours, minArea=2.)
    groups = groupBlobs(blobs, maxDist=150.)

    # Draw the log(intensity) of the pixel values
    #if args.log:
    #    image = np.log10(1. + image)

    # Plot the image using matplotlib
    mpl.rc("font", family="serif", size=14)

    title = os.path.basename(path)
    # Zoom in on grouped blobs in separate figures
    for i, bg in enumerate(groups):
        X0, X1, Y0, Y1 = bg.getSquareBoundingBox()
        l1, l2, theta = bg.getPrincipalMoments(image)

        fig = plt.figure(figsize=(6,6))
        axg = fig.add_subplot(111)
        im = axg.imshow(image, cmap=mpl.cm.hot,
                        interpolation="nearest", aspect="auto",
                        extent=[x0, x1, y0, y1])
        for blob in bg.blobs:
            axg.plot(blob.x, blob.y, linewidth=2, color="#00dd00")
        axg.set_xlim([X0-5, X1+5])
        axg.set_ylim([Y0-5, Y1+5])
        axg.set_xlabel("pixels")
        axg.set_ylabel("pixels")
        axg.set_title("%s: Cluster %d" % (title, i+1))
    plt.show()

# Import all needed python modules
try:
    import argparse, os, sys, datetime
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from skimage import measure
    from PIL import Image
except ImportError,e:
    print e
    raise SystemExit

# Open file containing test paths
f = open('./test_image_paths', 'r+')

# Open user log file
# Note: replace with regex
ctime = '_'.join(str(datetime.datetime.now()).split(' '))
x = open('./log_file_' + ctime, 'w')

possible = [1,2,3,4]
ct = -1
errorTolerance = 10
""" Loop through images and display while filtering out noise. Get user
input, hopefully a number, and compare to machine output then display
result. """
for line in f:
    ct+=1
    try:
        mtype = line.split(',')[1]
    except IndexError as _:
        print("Line could not be read.")
        errorTolerance -= 1
        if (errorTolerance < 1):
            raise SystemExit
    if line[0] == '/' and not mtype == 'noise' and not mtype == 'null':
        output(str(datetime.datetime.now()))
        path = line.split(',')[0]
        path_str = "path: " + path
        print(path_str)
        output(path_str)
        showFigure(path)
        uinput = int()
        while not uinput or not uinput in possible:
            print("Enter the highest number type (1: spot, 2: other, 3: worm, 4: track):")
            itype = raw_input()
            try:
                uinput = int(itype)
                if uinput in possible:
                    output("Human: " + type_of(uinput))
                    # Uses variable from above that declares machine event type
                    output("Machine: " + mtype)
                else:
                    print("Not a valid input")
            except ValueError, _:
                print("Invalid literal for int()")
        usrp = type_of(uinput)
        if (usrp == mtype.strip().lower()):
            result = "Same"
        else:
            result = "Different"
        print(result)
        output("Class: " + result)
    else:
        txct = "Bad line: " + str(ct)
        print(txct)
        output(txct)
    output(str())

# Close files
f.close() and x.close()

print("Thank you for your time. The results recorded will be used to assist\nwith the identification of cosmic rays at WIPAC (Press Enter)")

# Wait for user
raw_input()
os.system('clear && printf "\e[2t"')
raise SystemExit
# (c)2015 WIPAC
