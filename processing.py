import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize


def load_image_as_gray(image_file, max_size=512):
    image = Image.open(image_file).convert("L")
    arr = np.array(image)

    if max(arr.shape) > max_size:
        scale = max_size / max(arr.shape)
        new_shape = (int(arr.shape[0] * scale), int(arr.shape[1] * scale))
        arr = resize(arr, new_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    return arr


def quantize_gray_image(gray_image, levels):
    gray_image = gray_image.astype(np.float32)
    quantized = np.floor(gray_image / 256 * levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)
    return quantized


def compute_glcm_features(gray_image, levels=16, distance=1, angle_degrees=0):
    angle_radians = np.deg2rad(angle_degrees)
    quantized = quantize_gray_image(gray_image, levels)

    glcm = graycomatrix(
        quantized,
        distances=[distance],
        angles=[angle_radians],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    features = {
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
        "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
    }

    matrix = glcm[:, :, 0, 0]

    return quantized, matrix, features

def scale_quantized_for_display(quantized, levels):
    return (quantized.astype(float) / (levels - 1) * 255).astype("uint8")