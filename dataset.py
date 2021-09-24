# coding: utf-8

from typing import Dict, Tuple, List

from torchvision.datasets import ImageFolder
from torchvision import transforms


class HogweedClassificationDataLoader(ImageFolder):

    def __init__(self, root: str = "prepared_data/images_train", *args, **kwargs):
        super(HogweedClassificationDataLoader, self).__init__(root, *args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ["no_hogweed", "has_hogweed"], {"no_hogweed": 0, "has_hogweed": 1}


if __name__ == "__main__":

    data_loader = HogweedClassificationDataLoader(root="prepared_data/images_train",
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Resize(750)]))
    tensor, label = data_loader.__getitem__(0)

    print("Label:", label)
    print("Tensor shape:", tensor.shape)
