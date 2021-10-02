# coding: utf-8
import os

from torchvision import transforms
from tqdm import tqdm

from dataset import HogweedClassificationDataset


def example(dataset, i):
    """ Preparing an image for viewing or saving """
    return transforms.ToPILImage()(dataset[i][0])


SHORT_SIDE = 300

train_set = HogweedClassificationDataset(root="prepared_data/images_train",
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Resize(SHORT_SIDE)]))

try:
    os.mkdir("prepared_data/images_train_resized")
except:
    pass

try:
    os.mkdir("prepared_data/images_train_resized/has_hogweed")
except:
    pass

try:
    os.mkdir("prepared_data/images_train_resized/no_hogweed")
except:
    pass

for idx, (image_path, image_label) in tqdm(enumerate(train_set.samples)):
    example(train_set, idx).save(image_path.replace("images_train", "images_train_resized"))


test_set = HogweedClassificationDataset(root="prepared_data/images_test",
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Resize(SHORT_SIDE)]))


try:
    os.mkdir("prepared_data/images_test_resized")
except:
    pass

try:
    os.mkdir("prepared_data/images_test_resized/unknown")
except:
    pass

for idx, (image_path, image_label) in tqdm(enumerate(train_set.samples)):
    example(test_set, idx).save(image_path.replace("images_test", "images_test_resized"))
