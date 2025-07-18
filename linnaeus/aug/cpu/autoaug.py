# linnaeus/aug/cpu/autoaug.py

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from linnaeus.aug.base import AutoAugmentBatch
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPUAutoAugmentBatch(AutoAugmentBatch):
    def __init__(self, policy: str, color_jitter: float, config=None):
        super().__init__(policy, color_jitter, config=config)
        self.ops = self._create_cpu_ops()

    # --- Step 1: Define each operation as a named method ---

    def _op_shear_x(self, img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, magnitude * 0.3, 0, 0, 1, 0))

    def _op_shear_y(self, img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.3, 1, 0))

    def _op_translate_x(self, img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] / 10, 0, 1, 0))

    def _op_translate_y(self, img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] / 10))

    def _op_rotate(self, img, magnitude):
        return img.rotate(magnitude)

    def _op_color(self, img, magnitude):
        return ImageEnhance.Color(img).enhance(1 + magnitude * 0.9)

    def _op_posterize(self, img, magnitude):
        return ImageOps.posterize(img, int(magnitude))

    def _op_solarize(self, img, magnitude):
        return ImageOps.solarize(img, 256 - int(magnitude))

    def _op_contrast(self, img, magnitude):
        return ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.9)

    def _op_sharpness(self, img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.9)

    def _op_brightness(self, img, magnitude):
        return ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.9)

    def _op_autocontrast(self, img, _):
        return ImageOps.autocontrast(img)

    def _op_equalize(self, img, _):
        return ImageOps.equalize(img)

    def _op_invert(self, img, _):
        return ImageOps.invert(img)

    def _op_solarize_add(self, img, magnitude):
        return self._solarize_add_helper(img, magnitude)

    def _op_posterize_original(self, img, magnitude):
        return ImageOps.posterize(img, int(magnitude))

    def _op_posterize_increasing(self, img, magnitude):
        return ImageOps.posterize(img, 8 - int(magnitude))

    def _op_desaturate(self, img, magnitude):
        return ImageEnhance.Color(img).enhance(1 - magnitude * 0.9)

    def _op_gaussian_blur_rand(self, img, magnitude):
        return img.filter(ImageFilter.GaussianBlur(radius=magnitude))

    # --- Step 2: Update _create_cpu_ops to reference the named methods ---
    def _create_cpu_ops(self) -> dict[str, callable]:
        """
        Create a dictionary of CPU-based augmentation operations by referencing
        the class's named methods, which are picklable.
        """
        ops = {
            "ShearX": self._op_shear_x,
            "ShearY": self._op_shear_y,
            "TranslateX": self._op_translate_x,
            "TranslateY": self._op_translate_y,
            "Rotate": self._op_rotate,
            "Color": self._op_color,
            "Posterize": self._op_posterize,
            "Solarize": self._op_solarize,
            "Contrast": self._op_contrast,
            "Sharpness": self._op_sharpness,
            "Brightness": self._op_brightness,
            "AutoContrast": self._op_autocontrast,
            "Equalize": self._op_equalize,
            "Invert": self._op_invert,
            "SolarizeAdd": self._op_solarize_add,
            "PosterizeOriginal": self._op_posterize_original,
            "PosterizeIncreasing": self._op_posterize_increasing,
            "Desaturate": self._op_desaturate,
            "GaussianBlurRand": self._op_gaussian_blur_rand,
        }
        return ops

    def __call__(self, images: np.ndarray) -> np.ndarray:
        augmented_images = []
        for img in images:
            img_uint8 = (img * 255).astype("uint8")
            pil_img = Image.fromarray(img_uint8)
            sub_policy = random.choice(self.policy)
            for op_name, prob, magnitude in sub_policy:
                if np.random.rand() < prob:
                    pil_img = self._apply_op(pil_img, op_name, magnitude)
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            augmented_images.append(img_array)
        result = np.stack(augmented_images)
        return result.astype(np.float32)

    def _solarize_add_helper(self, img: Image.Image, magnitude: float, threshold: int = 128) -> Image.Image:
        lut = []
        for i in range(256):
            if i < threshold:
                lut.append(min(255, i + magnitude))
            else:
                lut.append(i)
        if img.mode in ("L", "RGB"):
            if img.mode == "RGB" and len(lut) == 256:
                lut = lut + lut + lut
            return img.point(lut)
        else:
            return img

    def _apply_op(self, image: Image.Image, op_name: str, magnitude: int) -> Image.Image:
        if op_name not in self.ops:
            raise ValueError(f"Unknown operation: {op_name}")
        return self.ops[op_name](image, magnitude)
