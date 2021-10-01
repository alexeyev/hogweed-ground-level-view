# coding: utf-8
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset
from torchvision import transforms

from dataset import HogweedClassificationDataset
from plant_clef_resnet import load_plant_clef_resnet18


def example(dataset, i):
    """ Preparing an image for viewing or saving """
    print("oh", train_set[i][1])
    return transforms.ToPILImage()(dataset[i][0])


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
            target = target.to(device)
            output = model(data.to(device))
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":

    SEED = 100
    train_set = HogweedClassificationDataset(root="prepared_data/images_train_resized",
                                             transform=transforms.Compose([transforms.ToTensor()]))

    print("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)), train_set.targets,
        stratify=train_set.targets,
        test_size=0.15,
        shuffle=True,
        random_state=SEED
    )

    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices))

    print("CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up a model...")
    pretrained_resnet = load_plant_clef_resnet18(device=device)

    for ch_id, child in enumerate(pretrained_resnet.children()):
        for _, param in enumerate(child.parameters()):
            if param.requires_grad is not None:
                param.requires_grad = False

    pretrained_resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=2)
    )

    optimizer = optim.AdamW(pretrained_resnet.parameters(), amsgrad=True)
    loss_function = loss.CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(1, 100):
        train(pretrained_resnet, train_loader, optimizer, loss_function, epoch, device)
        test(pretrained_resnet, val_loader, loss_function, device)

    print("It is done.")
