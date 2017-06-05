from io import BytesIO
from matplotlib.pyplot import imshow
from IPython.display import display
import copy
import numpy as np
from PIL import Image, ImageOps  # PIL is the Python Imaging Library
import requests

def image_open(imagefile):
    with open(imagefile, "rb") as in_file:
        img = Image.open(BytesIO(in_file.read()))
        #imshow(np.asarray(img))
    return img

def image_save(filepath, img):
    with open(filepath, "w") as out_file:
        img.save(out_file)
    return

def identity_image(img):
    return img

def mirror_image(img):
    img_mirror = ImageOps.mirror(img)
    return img_mirror

# Crop the image to random coordinates and resize the image
def crop_image(img, i):
    w, h = img.size
    np.random.seed(4 * i)
    x0 = np.random.random() * (w / 4)
    y0 = np.random.random() * (h / 4)
    x1 = w - np.random.random() * (w / 4)
    y1 = h - np.random.random() * (h / 4)

    image_crop = img.crop([int(x0), int(y0), int(x1), int(y1)])
    image_crop = image_crop.resize((w, h), Image.BILINEAR)
    return image_crop

def compress_image(img):
    buffer = BytesIO()
    img.save(buffer, "jpeg", quality=5)
    buffer.seek(0)
    image_compressed = Image.open(buffer)
    return image_compressed

# Homography transformation
def homography_image(img):
    w, h = img.size
    data = (int(0.25 * w), 0, 0, h, w, h, int(0.75 * w), 0)
    image_homo = img.transform((w, h), Image.QUAD, data, Image.BILINEAR)
    return image_homo


def rotate_image(image, angle):
    for i in range(abs(angle)):
        white = Image.new('L', image.size, "white")
        wr = white.rotate(np.sign(angle), Image.NEAREST, expand=0)
        im = image.rotate(np.sign(angle), Image.BILINEAR, expand=0)
        image.paste(im, wr)
    return image

def scale_image(img):
    scale = 1.5 ** 4.5
    w, h = img.size
    scale_dimensions = int(w / scale), int(h / scale)
    image_scaled = img.resize(scale_dimensions, Image.BILINEAR)
    image_scaled = image_scaled.resize((w, h), Image.BILINEAR)
    return image_scaled

def color_image(image, i):
    lcolor = [381688.61379382, 4881.28307136, 2316.10313483]
    pcolor = [[-0.57848371, -0.7915924, 0.19681989],
              [-0.5795621, 0.22908373, -0.78206676],
              [-0.57398987, 0.56648223, 0.59129816]]
    # pre-generated gaussian values
    alphas = [[0.004894, 0.153527, -0.012182],
              [-0.058978, 0.114067, -0.061488],
              [0.002428, -0.003576, -0.125031]]
    p1r = pcolor[0][0]
    p1g = pcolor[1][0]
    p1b = pcolor[2][0]
    p2r = pcolor[0][1]
    p2g = pcolor[1][1]
    p2b = pcolor[2][1]
    p3r = pcolor[0][2]
    p3g = pcolor[1][2]
    p3b = pcolor[2][2]

    l1 = np.sqrt(lcolor[0])
    l2 = np.sqrt(lcolor[1])
    l3 = np.sqrt(lcolor[2])

    if i <= 2:
        alpha = alphas[i]
    else:
        np.random.seed(i * 3)
        alpha = np.random.randn(3, 0, 0.01)
    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]

    (dr, dg, db) = (a1 * l1 * p1r + a2 * l2 * p2r + a3 * l3 * p3r,
                    a1 * l1 * p1g + a2 * l2 * p2g + a3 * l3 * p3g,
                    a1 * l1 * p1b + a2 * l2 * p2b + a3 * l3 * p3b)

    table = np.tile(np.arange(256), 3).astype(np.float64)
    table[:256] += dr
    table[256:512] += dg
    table[512:] += db
    image = image.convert("RGB").point(table)
    return image

def crop_corner_image(img, n):
    w, h = img.size
    x0 = 0
    x1 = w
    y0 = 0
    y1 = h

    rat = 256 - 227

    if n == 0:  # center
        x0 = (rat * w) / (2 * 256.0)
        y0 = (rat * h) / (2 * 256.0)
        x1 = w - (rat * w) / (2 * 256.0)
        y1 = h - (rat * h) / (2 * 256.0)
    elif n == 1:
        x0 = (rat * w) / 256.0
        y0 = (rat * h) / 256.0
    elif n == 2:
        x1 = w - (rat * w) / 256.0
        y0 = (rat * h) / 256.0
    elif n == 3:
        x1 = w - (rat * w) / 256.0
        y1 = h - (rat * h) / 256.0
    else:
        assert n == 4
        x0 = (rat * w) / 256.0
        y1 = h - (rat * h) / 256.0

    image_corner = img.crop((int(x0), int(y0), int(x1), int(y1)))
    image_corner = image_corner.resize((w, h), Image.BILINEAR)
    return image_corner

