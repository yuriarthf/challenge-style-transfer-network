#!/usr/bin/python3
import os
import argparse
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
from pathlib import Path
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(
        description="Style images by specifying an input image or diretory and style directory, It's also possible to "
                    "define an output dir, but if not, it will default to CURRENT_DIR/output.")
    parser.add_argument(
        "--input-images", "-i", dest="input_images",
        help="Input image(s) to be styled, could be a list.",
        nargs='*')
    parser.add_argument(
        "--input-dir", "-I", dest="input_dir", help="Input directory with the images to style.")
    parser.add_argument(
        "--style-images", "-s", dest="style_images",
        help="Style image(s) to be used.",
        nargs='*')
    parser.add_argument(
        "--style-dir", "-S", dest="style_dir", help="Style images' directory.", required=True)
    parser.add_argument(
        "--output-dir", "-o", dest="output_dir",
        help="Output directory (relative or absolute) for the styled images, defaults to CURRENT_DIR/output.",
        default="output")

    parser.add_argument(
        "--get-max-dim-from", "-d",
        help="Get max_dim parameters from the input images ('input'), style images ('style') or constant (--max-dim), "
             "defaults to 'constant'.", choices=["input", "style", "constant"], default="constant",
        dest="max_dim_from")
    parser.add_argument(
        "--max-dim",
        help="Sets a constant max_dim, applicable only when (--get-max-dim-from) is set to constant",
        type=int, default=1280, dest="max_dim")
    parser.add_argument(
        "--input-batch-size", "-b",
        help="Stylize images in batches of given length",
        type=int, default=None, dest="input_batch_size"
    )

    args = parser.parse_args()
    if not any((args.input_images, args.input_dir)):
        parser.error("--input-images (-i) or --input-dir (-I) must be provided.")
    if not any((args.style_images, args.style_dir)):
        parser.error("--style-images (-s) or --style-dir (-S) must be provided.")
    return args


class ImageStylization:

    # Image formats
    IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}

    def __init__(self, input_image_or_dir, style_image_or_dir, output_path, max_dim_from="constant",
                 model_url="https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2",
                 constant_dim=1280):

        if not all((input_image_or_dir, style_image_or_dir, output_path)):
            raise ValueError("<input_image_or_dir>, <style_dir> and <output_path> must be specified.")

        # Load compressed models from tensorflow_hub
        if not os.environ.get('TFHUB_MODEL_LOAD_FORMAT'):
            os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

        self.input_images = self._get_images_path(Path(input_image_or_dir))
        self.style_images = self._get_images_path(Path(style_image_or_dir))

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        if not os.access(self.output_path, os.W_OK):
            raise OSError("Directory must have write permissions for the current user.")

        self.hub_model = self._get_model(model_url)

        self.max_dim = self._get_max_dim(get_from=max_dim_from, constant_dim=constant_dim)

    def _get_max_dim(self, get_from="input", constant_dim=1280):
        if get_from not in {"input", "style", "constant"}:
            raise ValueError("<get_from> must be equal to 'input' or 'style'")
        if get_from == "constant":
            return constant_dim
        if get_from == "input":
            return self._get_images_max_dim(self.input_images)
        if get_from == "style":
            return self._get_images_max_dim(self.style_images)

    @staticmethod
    def _get_images_max_dim(image_paths):
        max_dim = 0
        for image_path in image_paths:
            image = PIL.Image.open(image_path)
            image_long_dim = max(image.size)
            if image_long_dim > max_dim:
                max_dim = image_long_dim
        return max_dim

    @staticmethod
    def _get_images_path(image_or_dir):
        image_paths = list()
        if image_or_dir.is_dir():
            for file in image_or_dir.iterdir():
                file.is_file() and (file.suffix in ImageStylization.IMAGE_FORMATS) and image_paths.append(file)
        else:
            image_paths.extend(image_or_dir if isinstance(image_or_dir, list) else [image_or_dir])
        return image_paths

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def load_img(self, path_to_img: Path, resize_img=True):
        img = tf.io.read_file(str(path_to_img))
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if resize_img:
            shape = tf.cast(tf.shape(img)[:-1], tf.float32)
            long_dim = max(shape)
            scale = self.max_dim / long_dim
            new_shape = tf.cast(shape * scale, tf.int32)
            img = self._resize_img(img, new_shape)
        return img

    def _get_imgs_max_shape(self, image_list):
        max_size = 0
        max_shape = None
        for image in image_list:
            shape = tf.cast(tf.shape(image)[:-1], tf.float32)
            long_dim = max(shape)
            scale = self.max_dim / long_dim
            new_shape = tf.cast(shape * scale, tf.int32)
            if max_shape is None:
                max_shape = new_shape
                continue
            if np.product(new_shape) > max_size:
                max_size = np.product(new_shape)
                max_shape = new_shape
        return max_shape

    @staticmethod
    def _resize_img(img, new_shape):
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def _get_model(url):
        return hub.load(url)

    def _style_images(self, input_image_paths, input_style_path, model):
        image_list = list()
        for input_image_path in input_image_paths:
            image_list.append(tf.constant(self.load_img(input_image_path, resize_img=False)))
        max_shape = self._get_imgs_max_shape(image_list)
        image_batches = tf.concat(tuple(map(lambda img: self._resize_img(img, max_shape), image_list)), axis=0)
        style = tf.constant(self.load_img(input_style_path))
        images = list()
        for image_tensor in model(image_batches, style)[0]:
            images.append(self.tensor_to_image(image_tensor))
        return images

    def _style_image(self, input_image_path: Path, input_style_path: Path, model):
        image = self.load_img(input_image_path)
        style = self.load_img(input_style_path)
        return self.tensor_to_image(model(tf.constant(image), tf.constant(style))[0])

    def _save_image_to_dir(self, styled_image, input_image_path: Path, input_style_path: Path, img_format="png"):
        image_name = Path(input_image_path.stem + "_" + input_style_path.stem)
        styled_image.save(fp=(self.output_path / image_name), format=img_format)

    def process_images(self, input_batch=None):
        if input_batch:
            input_batches = list()
            begin_idx = 0
            end_idx = input_batch
            while begin_idx < len(self.input_images):
                if end_idx <= len(self.input_images):
                    input_batches.append(self.input_images[begin_idx:end_idx])
                else:
                    input_batches.append(self.input_images[begin_idx:])
                begin_idx = end_idx
                end_idx += input_batch
            for input_batch, style in product(input_batches, self.style_images):
                styled_images = self._style_images(input_batch, style, self.hub_model)
                for image, styled_image in zip(input_batch, styled_images):
                    self._save_image_to_dir(styled_image, image, style)
        else:
            for image, style in product(self.input_images, self.style_images):
                styled_image = self._style_image(image, style, self.hub_model)
                self._save_image_to_dir(styled_image, image, style)


def main():
    args = parse_args()

    image_stylization = ImageStylization(input_image_or_dir=args.input_images or args.input_dir,
                                         style_image_or_dir=args.style_images or args.style_dir,
                                         output_path=args.output_dir,
                                         max_dim_from=args.max_dim_from,
                                         constant_dim=args.max_dim)
    image_stylization.process_images(input_batch=args.input_batch_size)


if __name__ == "__main__":
    main()
