from tools import Inference
from omegaconf import OmegaConf
import os
import sys
import PIL
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf


def predict(comp, mask):
    comp = comp.convert("RGB")
    mask = mask.convert("1")
    in_shape = comp.size[::-1]
    image_size = args.input.image_size
    comp = tf.resize(comp, [image_size, image_size])
    mask = tf.resize(mask, [image_size, image_size])

    compt = tf.to_tensor(comp)
    maskt = tf.to_tensor(mask)
    res = infer.harmonize(compt, maskt)
    res = tf.resize(res, in_shape)

    return np.uint8((res * 255)[0].permute(1, 2, 0).numpy())


if __name__ == "__main__":
    args = OmegaConf.load(
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), f"config/test.yaml")
    )

    infer = Inference(**args)
    comp = PIL.Image.open(args.input.composite_path)
    mask = PIL.Image.open(args.input.mask_path)

    harmonized = predict(comp, mask)
    plt.imsave(args.output.path, harmonized)
