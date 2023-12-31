import math
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import cv2
from torchvision import transforms, models
from PIL import Image
import torchvision.transforms.functional as tf

# --------------------------------------------Metric tools-------------------------------------------- #


def lab_shift(x, invert=False):
    x = x.float()
    if invert:
        x[:, 0, :, :] /= 2.55
        x[:, 1, :, :] -= 128
        x[:, 2, :, :] -= 128
    else:
        x[:, 0, :, :] *= 2.55
        x[:, 1, :, :] += 128
        x[:, 2, :, :] += 128

    return x


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_fpsnr(fmse):
    return 10 * math.log10(255.0 / (fmse + 1e-8))


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), bit=8):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    norm = float(2**bit) - 1
    # print('before', tensor[:,:,0].max(), tensor[:,:,0].min(), '\t', tensor[:,:,1].max(), tensor[:,:,1].min(), '\t', tensor[:,:,2].max(), tensor[:,:,2].min())
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # print('clamp ', tensor[:,:,0].max(), tensor[:,:,0].min(), '\t', tensor[:,:,1].max(), tensor[:,:,1].min(), '\t', tensor[:,:,2].max(), tensor[:,:,2].min())
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * norm).round()
    return img_np.astype(out_type)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The input RGB image is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to RGB.

    The L channel is assumed to be in the range of :math:`[0, 100]`.
    a and b channels are in the range of :math:`[-128, 127]`.

    Args:
        image: Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The output RGB image are in the range of :math:`[0, 1]`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = 3.2404813432005266 * x + - \
        1.5371515162713185 * y + -0.4985363261688878 * z
    g: torch.Tensor = -0.9692549499965682 * x + \
        1.8759900014898907 * y + 0.0415559265582928 * z
    b: torch.Tensor = 0.0556466391351772 * x + - \
        0.2040413383665112 * y + 1.0573110696453443 * z

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(
        ((image + 0.055) / 1.055), 2.4), image / 12.92)

    return lin_rgb


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image: linear RGB Image to be converted to sRGB of shape :math:`(*,3,H,W)`.

    Returns:
        sRGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        image > threshold, 1.055 *
        torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image
    )

    return rgb


# --------------------------------------------Inference tools-------------------------------------------- #
def inference_img(model, img, device='cpu'):
    h, w, _ = img.shape
    # print(img.shape)
    if h % 8 != 0 or w % 8 != 0:
        img = cv2.copyMakeBorder(img, 8-h % 8, 0, 8-w %
                                 8, 0, cv2.BORDER_REFLECT)
    # print(img.shape)

    tensor_img = torch.from_numpy(img).permute(2, 0, 1).to(device)
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
    with torch.no_grad():
        out = model(input_t)
    # print("out",out.shape)
    result = out[0][:, -h:, -w:].cpu().numpy()
    # print(result.shape)

    return result[0]


def log(msg, lvl='info'):
    if lvl == 'info':
        print(f"***********{msg}****************")
    if lvl == 'error':
        print(f"!!! Exception: {msg} !!!")


def harmonize(comp, mask, model):
    log("Inference started")
    if comp is None or mask is None:
        log("Empty source")
        return np.zeros((16, 16, 3))

    comp = comp.convert('RGB')
    mask = mask.convert('1')
    in_shape = comp.size[::-1]

    comp = tf.resize(comp, [model.image_size, model.image_size])
    mask = tf.resize(mask, [model.image_size, model.image_size])

    compt = tf.to_tensor(comp)
    maskt = tf.to_tensor(mask)
    res = model.harmonize(compt, maskt)
    res = tf.resize(res, in_shape)

    log("Inference finished")

    return np.uint8((res*255)[0].permute(1, 2, 0).numpy())


def extract_matte(img, back, model):
    mask, fg = model.extract(img)
    fg_pil = Image.fromarray(np.uint8(fg))

    composite = fg + (1 - mask[:, :, None]) * \
        np.array(back.resize(mask.shape[::-1]))
    composite_pil = Image.fromarray(np.uint8(composite))

    return [composite_pil, mask, fg_pil]


def css(height=3, scale=2):
    return f".output_image {{height: {height}rem !important; width: {scale}rem !important;}}"
