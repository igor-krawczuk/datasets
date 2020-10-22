"""
Adapted from https://www.tensorflow.org/datasets/catalog/tf_flowers

@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }
"""

URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
from torch.utils.data import Dataset
import os
import subprocess as sp
from PIL import Image
import numpy as np
from filelock import FileLock
from tempfile import mktemp


def download_and_unpack(URL):
    fname = os.path.basename(URL)
    with FileLock("/tmp/flowers_download"):
        print(f"Downloading into {os.path.abspath(os.curdir)}")
        if not os.path.exists(fname):
                sp.call(["wget", URL])
        if not os.path.exists("flower_photos"):
            sp.call(["tar", "xf", fname])


class TFFlowers(Dataset):
    def __init__(
            self, data_dir=os.path.expanduser("~/.datasets/tfflowers"), img_size=(240, 240)
    ):
        super().__init__()
        data_dir = os.path.abspath(data_dir)
        cwd = os.path.abspath(os.curdir)
        self.img_size = img_size
        os.makedirs(data_dir, exist_ok=True)
        os.chdir(data_dir)
        download_and_unpack(URL)
        self.data_dir = os.path.join(data_dir, "flower_photos")
        self.ind_to_class = {
            i: v
            for i, v in enumerate(
                ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
            )
        }
        self.class_to_ind = {i: v for v, i in self.ind_to_class.items()}
        self.classes = []

        self.paths = []
        for k in self.class_to_ind.keys():
            k_paths = [
                os.path.abspath(os.path.join(self.data_dir, k, x))
                for x in os.listdir(os.path.join(self.data_dir, k))
            ]
            c = self.class_to_ind[k]
            for i, p in enumerate(k_paths):
                self.paths.append(p)
                self.classes.append(c)
        os.chdir(cwd)
        self.classes = np.array(self.classes)
        self.paths = np.array(self.paths)

    def __getitem__(self, item):
        c = self.classes[item]
        path = self.paths[item]
        with Image.open(path) as img:
            img: Image
            img = np.asarray(img.resize(self.img_size))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img, c

    def __len__(self):
        return len(self.classes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = TFFlowers(data_dir="/tmp/flowers")
    shapes = [x.shape for (x, _) in ds]
    print(set(shapes))
