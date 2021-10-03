# coding: utf-8
import torch
from torchvision import models


def load_plant_clef_resnet18(path: str = "plants_epoch_100/model_best.pth.tar",
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
        The trained model can be obtained using this instruction
        https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-plants.md
        and this: https://nvidia.box.com/s/dslt9b0hqq7u71o6mzvy07w0onn0tw66

        Model and used code are under NVIDIA license; Dustin F., you are the best.
    """

    # load the model checkpoint
    checkpoint = torch.load(path, map_location=device)
    arch = checkpoint['arch']

    # create the model architecture
    print('using model:  ' + arch)
    model = models.__dict__[arch](pretrained=True)

    # reshape the model's output
    model.fc = torch.nn.Linear(model.fc.in_features, checkpoint["num_classes"])

    # load the model weights
    model.load_state_dict(checkpoint['state_dict'])

    return model