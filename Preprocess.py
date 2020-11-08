import os
from collections import Counter

import cv2
import numpy as np

TILE_WIDTH = 25
TILE_HEIGHT = 25
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # starting kernel from OpenCV example
BLUR_KERNEL = (85, 85)
# TODO: Scale these kernels based on image size


'''
    preprocesses imageFilepath image file
    returns filename of processed image, and a boolean:
        True indicates sharpening (presence of face possible from skin tones), 
        False indicates blurring (no skin tones detected)
    
    Example:
    import preprocessFile as pp
    pathToProcessedImage, isSharpened = pp.preprocess(image1.jpg)
    // run VJ or MTCNN using pathToProcessedImage

'''
def preprocess(imageFilepath):
    img = cv2.imread(imageFilepath) # IMPORTANT: OpenCV reads as B,G,R, not RGB, as far as I can tell
    tiles = splitImageIntoTiles(img, TILE_WIDTH, TILE_HEIGHT)
    name, extension = os.path.splitext(imageFilepath)

    imageContainsTileThatHasSkinAsDominantColour = False
    for tile in tiles:
        mostFreqColour = getMostFrequentColourTuple(tile)
        if determineIfBGRFallsWithinYCbCrRange(mostFreqColour):
            imageContainsTileThatHasSkinAsDominantColour = True
            break

    if imageContainsTileThatHasSkinAsDominantColour:
        # Sharpen the image, as skin might indicate a face
        sharpened = cv2.filter2D(img, -1, SHARPEN_KERNEL)
        # Write out
        newName = name + '_T' + extension
        cv2.imwrite(newName, sharpened)
        print(newName + " Detected skin colour")
        return newName, True
    else:
        # Blur the image, doesn't look like there will be a face
        blurred = cv2.blur(img, BLUR_KERNEL)
        # Write out
        newName = name + '_F' + extension
        cv2.imwrite(newName, blurred)
        print(newName + " DID NOT detect skin colour")
        return newName, False


'''
    pass image array and integers specifying desired width and height of tiles.
    returns a numpy array of tiles, each element with shape (height, width, 3)
'''
def splitImageIntoTiles(image, widthOfTile, heightOfTile):
    # Nice one liner taken from. It should not crop out/miss areas, they will be smaller tiles at the edges.
    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    tiles = [image[x:x + heightOfTile, y:y + widthOfTile] for x in range(0, image.shape[0], heightOfTile)
             for y in range(0, image.shape[1], widthOfTile)]
    # For debugging...
    # for index, tile in enumerate(tiles):
    #     cv2.imwrite('faceTest' + str(index) + '.png', tile)
    # cv2.imwrite('faceTestOrig.png', face)
    return tiles

# Implementation one, use the simple counter as I originally thought of
# Unfortunately, this might not work well in practice, as in a 25x25 square, it might only have a handful of pixels
# as a common colour leading to a bad dominant reading
def getMostFrequentColourTuple(image):
    tileList=image.reshape(-1, image.shape[-1]).tolist()
    for index, pixel in enumerate(tileList):
        tileList[index] = tuple(pixel) # Change to tuples for hashability

    colourList = Counter(tileList)
    return max(colourList, key=colourList.get)


'''
Returns a boolean indicating if the bgr tuple falls within our defined range in YCrCb
'''
def determineIfBGRFallsWithinYCbCrRange(bgrTuple):
    # Convert RGB to YCbCr.
    R = bgrTuple[2]
    G = bgrTuple[1]
    B = bgrTuple[0]
    # Manual transformation: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    # This is full range YCrCb, some online converters don't seem to use truncated one for TVs
    Y = int(round(0.299 * R + 0.587 * G + 0.114 * B))
    Cr = int(round((R - Y) * 0.713 + 128))
    Cb = int(round((B - Y) * 0.564 + 128))
    return (60 <= Y <= 255) and (155 <= Cr <= 180) and (65 <= Cb <= 105)

# Use a unit test framework? Nah...
# I hope this generates correct test cases: http://www.picturetopeople.org/color_converter.html
# Nope... it does not according to the math... interesting...
# assert(determineIfBGRFallsWithinYCbCrRange((123, 211, 234)) == (195, 86, 144))


# Implementation 2
# I think a more proper way is to generate a histogram and use k-clustering to determine the dominant colour

# Demo two files below
# preprocess('detail2.png')
# preprocess('detail3.png')


def getYCrCbMask(img):
    lower = np.array([60, 135, 85])
    upper = np.array([255, 180, 135])
    # mask1 = cv2.inRange(img, 60, 255)
    # mask2 = cv2.inRange(img, 135, 180)
    # mask3 = cv2.inRange(img, 85, 135)
    # mask = mask1 + mask2 + mask3
    mask = cv2.inRange(img, lower, upper)
    return mask

def getYCrCbMask2(img):
    lower = np.array([60, 155, 65])
    upper = np.array([255, 180, 105])
    mask = cv2.inRange(img, lower, upper)
    return mask

def getYCrCbMask3(img):
    lower = np.array([60, 145, 65])
    upper = np.array([255, 180, 115])
    mask = cv2.inRange(img, lower, upper)
    return mask

def maskImgAndOutRGBImg(img):
    img = cv2.imread(img)
    # mask = getYCrCbMask(img)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    mask = getYCrCbMask(img_YCrCb)
    mask2 = getYCrCbMask2(img_YCrCb)
    mask3 = getYCrCbMask3(img_YCrCb)
    # img_YCrCb[np.where(mask == 0)] = 0
    maksedImg = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask)
    maksedImg2 = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask2)
    maksedImg3 = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask3)
    img_rgb = cv2.cvtColor(maksedImg, cv2.COLOR_YCR_CB2BGR)
    img_rgb2 = cv2.cvtColor(maksedImg2, cv2.COLOR_YCR_CB2BGR)
    img_rgb3 = cv2.cvtColor(maksedImg3, cv2.COLOR_YCR_CB2BGR)
    # Background is green due to bit operations in YCrCb space
    # lower = np.array([0, 0, 0])
    # upper = np.array([0, 134, 0])
    # mask4 = cv2.inRange(img_rgb, lower, upper)
    # lower = np.array([0, 136, 0])
    # upper = np.array([0, 255, 0])
    # mask5 = cv2.inRange(img_rgb, lower, upper)
    # mask6 = mask4 + mask5
    # maskedRGB = cv2.bitwise_and(img_rgb, img_rgb, mask=mask6)
    # cv2.imwrite('0002_1_M.jpg', maskedRGB)
    cv2.imwrite('0002_1_M.jpg', img_rgb)
    cv2.imwrite('0002_1_M_2.jpg', img_rgb2)
    cv2.imwrite('0002_1_M_3.jpg', img_rgb3)

def maskImgAndOutRGBImg2(img):
    img = cv2.imread(img)
    # mask = getYCrCbMask(img)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    mask = getYCrCbMask(img_YCrCb)
    mask2 = getYCrCbMask2(img_YCrCb)
    mask3 = getYCrCbMask3(img_YCrCb)
    # img_YCrCb[np.where(mask == 0)] = 0
    maksedImg = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask)
    maksedImg2 = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask2)
    maksedImg3 = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=mask3)
    img_rgb = cv2.cvtColor(maksedImg, cv2.COLOR_YCR_CB2BGR)
    img_rgb2 = cv2.cvtColor(maksedImg2, cv2.COLOR_YCR_CB2BGR)
    img_rgb3 = cv2.cvtColor(maksedImg3, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite('000008_M.jpg', img_rgb)
    cv2.imwrite('000008_M_2.jpg', img_rgb2)
    cv2.imwrite('000008_M_3.jpg', img_rgb3)

# maskImgAndOutRGBImg('0002_1.jpg')
# maskImgAndOutRGBImg2('000008.jpg')