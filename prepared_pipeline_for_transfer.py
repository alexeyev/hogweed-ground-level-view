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


def train(model, train_loader, optimizer, loss_function, current_epoch_number=0, device=None):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_function(output, target.unsqueeze(dim=-1).float().to(device))

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
            target = target.float().unsqueeze(dim=-1).to(device)
            output = model(data.to(device))
            test_loss += loss_function(output, target).sum().item()
            pred = (torch.sign(output - 0.5) + 1) / 2
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

    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices), batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices), shuffle=False, batch_size=128)

    print("CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up a model...")
    pretrained_resnet = load_plant_clef_resnet18(device=device)

    for ch_id, child in enumerate(pretrained_resnet.children()):
        for _, param in enumerate(child.parameters()):
            param.requires_grad = False

    pretrained_resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=1)
    )

    pretrained_resnet = pretrained_resnet.to(device)

    optimizer = optim.AdamW(pretrained_resnet.parameters())
    loss_function = loss.BCEWithLogitsLoss()

    print("Starting training...")

    for epoch in range(1, 100):
        train(pretrained_resnet, train_loader, optimizer, loss_function, epoch, device)
        test(pretrained_resnet, val_loader, loss_function, device)

        if epoch == 5:
            for ch_id, child in enumerate(pretrained_resnet.children()):
                for _, param in enumerate(child.parameters()):
                    param.requires_grad = True

    print("It is done.")
