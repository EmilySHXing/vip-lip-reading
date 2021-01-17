from PIL import Image


def imgToGray(filename):
    """
    Jan.21st Shuhao Xing
    Given an input colored image, return the image object with grayscaled values
    :param filename: source file name of the input image
    :return: img: an image object after grayscaling
    """
    img = Image.open(filename)
    pixels = img.load()  # load the pixels of the input image
    for row in range(0,img.size[0]):
        for column in range(0,img.size[1]):  # loop through every single pixel in the image
            R = pixels[row, column][0]
            G = pixels[row, column][1]
            B = pixels[row, column][2]
            gray = int(0.3*R+0.59*G+0.11*B)  # compute the grayscale value using luminosity formula
            pixels[row, column] = (gray,gray,gray)
    return img


def smoothFilter(infile,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a smoothed image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: None
    """
    img = Image.open(infile)
    pixels = img.load()  # load the pixels of the input image
    k = 2
    avg = list()
    w,h = img.size
    for i in range(0,w):
        for j in range(0,h):  # loop through every single pixel in the image
            sumR = 0
            sumG = 0
            sumB = 0
            count = 0
            for m in range(max(0,i-k),min(w-1,i+k+1)):
                for n in range(max(0,j-k),min(h-1,j+k+1)):  # averages pixel values in 3x3 area around current pixel
                    sumR += pixels[m,n][0]
                    sumG += pixels[m,n][1]
                    sumB += pixels[m,n][2]
                    count += 1
            avg += [(int(sumR / count), int(sumG / count), int(sumB / count))]
    ind = 0
    for i in range(0,w):
        for j in range(0,h):
            pixels[i,j] = avg[ind]  # modify to the original pixel values
            ind += 1
    img.save(outfile)


def sharpFilter(infile,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a sharpened image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: none
    """
    img = Image.open(infile)
    pixels = img.load()  # load the pixels of the input image
    new = list()
    w, h = img.size
    for i in range(0, w):
        for j in range(0, h):  # loop through every single pixel in the image
            sumR = 0
            sumG = 0
            sumB = 0
            count = 0
            for m in range(max(0, i - 1), min(w - 1, i + 2)):
                for n in range(max(0, j - 1), min(h - 1, j + 2)):
                    sumR += pixels[m, n][0]
                    sumG += pixels[m, n][1]
                    sumB += pixels[m, n][2]
                    count += 1
            R = (count + 1) * pixels[i,j][0] - sumR  # convolution with matrix [[-1 -1 -1] [-1 8 -1] [-1 -1 -1]]
            G = (count + 1) * pixels[i,j][1] - sumG
            B = (count + 1) * pixels[i,j][2] - sumB
            new += [(R,G,B)]
    ind = 0
    for i in range(0, w):
        for j in range(0, h):
            pixels[i, j] = new[ind]  # update pixel values in original image
            ind += 1
    img.save(outfile)


def diffFilter(infile,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a differentiated image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: none
    """
    #img = imgToGray(infile)
    img = Image.open(infile)
    pixels = img.load()  # load the pixels of the input image
    w, h = img.size
    newPixels = list()
    for i in range(1, w-1):
        for j in range(1, h-1):  # loop through every single pixel in the image
            R = pixels[i, j][0] * (-8) + pixels[i, j - 1][0] + pixels[i, j + 1][0] + pixels[i - 1, j][0] + \
                pixels[i + 1, j][0] + pixels[i - 1, j - 1][0] + pixels[i + 1, j - 1][0] + pixels[i - 1, j + 1][0] + \
                pixels[i + 1, j + 1][0]
            G = pixels[i, j][0] * (-8) + pixels[i, j - 1][1] + pixels[i, j + 1][1] + pixels[i - 1, j][1] + \
                pixels[i + 1, j][1] + pixels[i - 1, j - 1][1] + pixels[i + 1, j - 1][1] + pixels[i - 1, j + 1][1] + \
                pixels[i + 1, j + 1][1]
            B = pixels[i, j][2] * (-8) + pixels[i, j - 1][2] + pixels[i, j + 1][2] + pixels[i - 1, j][2] + \
                pixels[i + 1, j][2] + pixels[i - 1, j - 1][2] + pixels[i + 1, j - 1][2] + pixels[i - 1, j + 1][2] + \
                pixels[i + 1, j + 1][2]
            newPixels += [(R,G,B)]  # convolution with matrix [[1 1 1][1 8 1][1 8 1]]
    ind = 0
    for i in range(1, w-1):
        for j in range(1, h-1):
            pixels[i,j] = newPixels[ind]  # update pixel values in original image
            ind += 1
    img.save(outfile)


def verFilter(infile,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a vertically edge detected image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: none
    """
    img = imgToGray(infile)
    pixels = img.load()  # load the pixels of the input image
    w, h = img.size
    newPixels = list()
    for i in range(1, w-1):
        for j in range(1, h-1):  # loop through every single pixel in the image
            R = - pixels[i - 1, j - 1][0] - 2 * pixels[i - 1, j][0] - pixels[i - 1, j + 1][0] + pixels[i + 1, j - 1][0] + \
                2 * pixels[i + 1, j][0] + pixels[i + 1, j + 1][0]
            G = - pixels[i - 1, j - 1][1] - 2 *pixels[i - 1, j][1] - pixels[i - 1, j + 1][1] + pixels[i + 1, j - 1][1] + \
                2 * pixels[i + 1, j][1] + pixels[i + 1, j + 1][1]
            B = - pixels[i - 1, j - 1][2] - 2 *pixels[i - 1, j][2] - pixels[i - 1, j + 1][2] + pixels[i + 1, j - 1][2] + \
                2 * pixels[i + 1, j][2] + pixels[i + 1, j + 1][2]
            newPixels += [(R,G,B)]  # applies vertical edge detection filter and stores the new value
    ind = 0
    for i in range(1, w-1):
        for j in range(1, h-1):
            pixels[i,j] = newPixels[ind]  # update pixel values in original image
            ind += 1
    img.save(outfile)

def horFilter(infile,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a horizontally edge detected image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: none
    """
    img = imgToGray(infile)
    pixels = img.load()  # load the pixels of the input image
    w, h = img.size
    newPixels = list()
    for i in range(1, w-1):
        for j in range(1, h-1):  # loop through every single pixel in the image
            R = pixels[i - 1, j - 1][0] + 2 * pixels[i, j - 1][0] + pixels[i + 1, j][0] - pixels[i - 1, j + 1][0] - \
                2 * pixels[i, j + 1][0] - pixels[i + 1, j + 1][0]
            G = pixels[i - 1, j - 1][1] + 2 * pixels[i, j - 1][1] + pixels[i + 1, j][1] - pixels[i - 1, j + 1][1] - \
                2 * pixels[i, j + 1][1] - pixels[i + 1, j + 1][1]
            B = pixels[i - 1, j - 1][2] + 2 * pixels[i, j - 1][2] + pixels[i + 1, j][2] - pixels[i - 1, j + 1][2] - \
                2 * pixels[i, j + 1][2] - pixels[i + 1, j + 1][2]
            newPixels += [(R,G,B)]  # applies horizontal edge detection filter and stores the new value
    ind = 0
    for i in range(1, w-1):
        for j in range(1, h-1):
            pixels[i,j] = newPixels[ind]  # update pixel values in original image
            ind += 1
    img.save(outfile)

def adapThres(infile,thres,outfile):
    """
    Jan.21th Shuhao Xing
    Given an input image, save a adapted binary image at specified source
    :param infile: source file name of the input image
    :param outfile: source file name of the output to be written to
    :return: none
    """
    img = imgToGray(infile)
    pixels = img.load()  # load the pixels of the input image
    w, h = img.size
    new = list()
    for i in range(0,w):
        for j in range(0,h):  # loop through every single pixel in the image
            sum = 0
            count = 0
            for m in range(max(0,i-1),min(w-1,i+2)):
                for n in range(max(0,j-1),min(h-1,j+2)):
                    sum += pixels[m,n][0]
                    count += 1
            avg = sum / count
            if pixels[i,j][0] <= (avg - thres):  # if the pixel value is less than threshold, set pixel to black
                new += [(0,0,0)]
            else:  # else set pixel to white
                new += [(255,255,255)]
    ind = 0
    for i in range(0, w):
        for j in range(0, h):
            pixels[i, j] = new[ind]  # update pixel values in original image
            ind += 1
    img.save(outfile)

if __name__ == "__main__":
    smoothFilter('test3.jpg','smooth3.jpg')
    sharpFilter('test3.jpg', 'sharp3.jpg')
    diffFilter('test3.jpg', 'diff3.jpg')
    verFilter('test3.jpg','v3.jpg')
    horFilter('test3.jpg','h3.jpg')