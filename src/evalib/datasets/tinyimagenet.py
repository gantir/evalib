"""
Author: Ramjee Ganti
Date: 2020-Apr-19
----
Original Author: Meng Lee, mnicnc404
Date: 2018/06/04
References:
    - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
"""

import glob
import os

from PIL import Image
from torch.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

EXTENSION = "JPEG"
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = "wnids.txt"
VAL_ANNOTATION_FILE = "val_annotations.txt"


class TinyImageNet(VisionDataset):

    """ TinyImageNet data set available from
    `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Args:
        root (string): Root directory including `train`, `test` and `val`
            subdirectories.
        split (string, optional): Indicating which split to return as a data set.
            Valid option: [`train`, `test`, `val`]
        transform (callable, optional): torchvision.transforms A (series) of valid
            transformation(s).
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        loader (callable, optional): A function to load an image given its path.
        in_memory: bool
            Set to True if there is enough memory (about 5G)
            and want to minimize disk IO overhead.

    Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        loader=None,
        in_memory=False,
        **kwargs
    ):

        super(VisionDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.split = split  # training set or validation set or test set

        if download:
            self.download()

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.%s" % EXTENSION), recursive=True
            )
        )
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == "test":
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.split
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


# if __name__ == "__main__":
# tiny_train = TinyImageNet('./dataset', split='train')
# print(len(tiny_train))
# print(tiny_train.__getitem__(99999))
# for fname, number in tiny_train.labels.items():
#     if number == 192:
#         print(fname, number)

# tiny_train = TinyImageNet('./dataset', split='val')
# print(tiny_train.__getitem__(99))

# in-memory test
# tiny_val = TinyImageNetData("dataset", split="val", in_memory=True)
