# coding: utf-8
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset
from torchvision import transforms

from dataset import HogweedClassificationDataset


def example(dataset, i):
    """ Preparing an image for viewing or saving """
    print("oh", train_set[i][1])
    return transforms.ToPILImage()(dataset[i][0])


def train(model, train_loader, optimizer, loss_function, current_epoch_number=0, device=None):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        print(batch_idx, ":", target)

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
    SHORT_SIDE = 30

    train_set = HogweedClassificationDataset(root="prepared_data/images_train",
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize(SHORT_SIDE)]))

    # print("Classes stats:", pd.Series(train_set.targets).value_counts())
    print(train_set.class_to_idx)
    # print(train_set.extensions)
    print(train_set.samples)
    # print(len(train_set.samples))
    # quit()

    # print("Saving transformed image sample...")
    # example(train_set, 10).save('out.jpg')

    print("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)), train_set.targets,
        stratify=train_set.targets,
        test_size=0.15,
        shuffle=True,
        random_state=SEED
    )

    # print(pd.Series(np.array(train_set.targets)[train_indices]).value_counts())
    # print(pd.Series(np.array(train_set.targets)[val_indices]).value_counts())

    # quit()

    # indices = list(range(len(train_set)))
    # split = int(0.15 * len(train_set))
    #
    # np.random.seed(SEED)
    # np.random.shuffle(indices)
    #
    # # todo: stratification?
    # train_idx, val_idx = indices[split:], indices[:split]
    #
    # print(pd.Series(np.array(train_set.targets)[train_idx]).value_counts())
    # print(pd.Series(np.array(train_set.targets)[val_idx]).value_counts())

    # train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices), batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices))

    print("CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting a model...")

    # a dummy model that doesn't work; should be replaced
    # with a pretrained model for finetuning
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), stride=(3, 3), padding=0),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=5, out_features=2)
    ).to(device)

    optimizer = optim.AdamW(model.parameters())
    loss_function = loss.CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(1, 5):
        train(model, train_loader, optimizer, loss_function, epoch, device)
        test(model, val_loader, loss_function, device)

    print("It is done.")
