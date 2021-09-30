# coding: utf-8

import torch
import torchvision
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import vgg16


def train(model, train_loader, optimizer, loss_function, current_epoch_number=0, device=None):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_function(output, target.to(device))

        # saving loss for stats
        loss_accum += loss.item() / len(data)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAveraged Epoch Loss: {:.6f}'.format(
                current_epoch_number,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_accum / (batch_idx + 1)))


def test(model, test_loader, loss_function, device):
    """ Testing an already trained model using the provided data from `test_loader` """

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            target = target.to(device)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":

    SEED = 100
    SHORT_SIDE = 600

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                               shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Setting a model...")

    vgg16_pretrained = vgg16(pretrained=True)
    vgg16_pretrained.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    vgg16_pretrained.classifier = nn.Sequential(
        nn.Linear(in_features=512, out_features=10),
        nn.Softmax(dim=-1)
    )

    for child in vgg16_pretrained.features[0:19]:
        for p in child.parameters():
            p.requires_grad = False

    vgg16_pretrained = vgg16_pretrained.to(device)

    optimizer = optim.Adam(vgg16_pretrained.parameters())
    loss_function = CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(1, 5):
        train(vgg16_pretrained, train_loader, optimizer, loss_function, epoch, device)
        test(vgg16_pretrained, test_loader, loss_function, device)

    print("It is done.")
