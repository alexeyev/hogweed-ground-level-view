# coding: utf-8

from typing import Dict, Tuple, List

from torchvision import transforms
from torchvision.datasets import ImageFolder


class HogweedClassificationDataset(ImageFolder):

    def __init__(self, root: str = "prepared_data/images_train", *args, **kwargs):
        super(HogweedClassificationDataset, self).__init__(root, *args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ["no_hogweed", "has_hogweed"], {"no_hogweed": 0, "has_hogweed": 1}


if __name__ == "__main__":
    data_loader = HogweedClassificationDataset(root="prepared_data/images_train",
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.RandomRotation((270, 270)),
                                                   transforms.Resize(750)]))
    tensor, label = data_loader.__getitem__(0)

    print("Label:", label)
    print("Tensor shape:", tensor.shape)
