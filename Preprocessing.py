import cv2
import numpy as np
from torchvision import transforms

from Startup import *


def transform_ndarray2tensor():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, resample=Image.BICUBIC),  # Arbitrary degree value
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # normalize the images to torchvision models specifications
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image

MAXIMUM_HEIGHT = 1024

MINIMUM_PARAM2 = 2
PARAM2_BASELINE = 10
BASELINE_AREA = 2588 * 1958

# Circular crop of the image
def load_twangy_color(path, image_size=IMG_SIZE, sigmaX=10):
    image = cv2.imread(path)
    height, width, _ = image.shape
    if height > MAXIMUM_HEIGHT:
        image = cv2.resize(image, (int(1024 * width/height), 1024))
        height, width, _ = image.shape
        assert False  # shouldn't happen on local machine since all the images are already rescaled
    param2 = max(MINIMUM_PARAM2, round(PARAM2_BASELINE * (width * height) / BASELINE_AREA))
    hough_image = create_binary_image(image, width, height)
    circles = cv2.HoughCircles(hough_image, cv2.HOUGH_GRADIENT, 1, 20, param1=20,
                               param2=param2, minRadius=int(height/3), maxRadius=int(width/1.5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if circles is None:
        # TODO better backup plan
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (image_size, image_size))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        print(f'Failed to find circles for {path}')
        # shouldn't fail (hasn't happened in like forever)
    else:
        circles = np.uint16(np.around(circles))
        circle = circles[0, 0]
        circle[2] *= 0.98   # TODO This is a really dumb idea
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])

        # figure out how much to crop
        x1 = x - r
        x2 = x + r + 1
        y1 = y - r
        y2 = y + r + 1

        # keep track of how much border to add
        left, right, top, bottom = 0, 0, 0, 0
        if x1 < 0:
            left = -x1
            x1 = 0
        if x2 > width:
            right = x2 - width
            x2 = width
        if y1 < 0:
            top = -y1
            y1 = 0
        if y2 > height:
            bottom = y2 - height
            y2 = height

        # crop it
        image = image[y1:y2, x1:x2, :]
        x = x - x1
        y = y - y1
        # pad the image with reflection of the image
        # TODO actually use a network to artificially generate this part to prevent fitting on metadata
        # TODO at least make it so some of the images aren't so weird
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
        x = x + left
        y = y + top

        # mask out the pixels outside the circle
        mask = np.zeros(image.shape[:-1], np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask)

        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.bitwise_not(create_binary_image(image, image_size, image_size, False))
        blurred = get_masked_blur(image, mask, sigmaX)
        image = cv2.addWeighted(image, 4, blurred, -4, 128)
        image[mask == 0] = 0

    return image


def load_preprocessed_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_binary_image(image, width, height, do_blur=True):
    hough_image = np.zeros((height + 2, width + 2), np.uint8)
    loDiff = 30
    upDiff = 10
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.floodFill(gray_image, hough_image, (0, 0),
                  None, loDiff=loDiff, upDiff=upDiff, flags=flags)
    cv2.floodFill(gray_image, hough_image, (0, height - 1),
                  None, loDiff=loDiff, upDiff=upDiff, flags=flags)
    cv2.floodFill(gray_image, hough_image, (width - 1, 0),
                  None, loDiff=loDiff, upDiff=upDiff, flags=flags)
    cv2.floodFill(gray_image, hough_image, (width - 1, height - 1),
                  None, loDiff=loDiff, upDiff=upDiff, flags=flags)
    hough_image = hough_image[1:height + 1, 1:width + 1]
    if do_blur:
        hough_image = cv2.GaussianBlur(hough_image, (7, 7), 0)
    return hough_image

def get_masked_blur(image, mask, sigmaX):
    image[mask == 0] = 0
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX)
    blurred_mask = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2RGB)
    blurred_mask[blurred_mask == 0] = 1
    result = 255. * blurred_image / blurred_mask
    result = np.rint(result).astype(np.uint8)
    return result
