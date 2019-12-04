import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch import optim
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore
import matplotlib.pyplot as plt

from networks import Net

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      transforms.Normalize((0.1307,), (0.3081,))
                                      ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                     ])

train_dataset = MNIST(root="./mnist", download=True, transform=train_transform)
test_dataset = MNIST(root="./mnist", train=False, download=True, transform=test_transform)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs...")
    net = nn.DataParallel(net)
net.to(device)

loss_fn = nn.CrossEntropyLoss()
lr = 0.003
optimizer = optim.Adam(net.parameters(), weight_decay=1e-6, lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# writer = SummaryWriter()


def compute_confusion_matrix(init_cm, y_true, y_pred):
    for i in range(len(y_true)):
        init_cm[y_true[i]][y_pred[i]] += 1


def train(epoch):
    net.train()
    total_loss = 0.
    for batch_idx, (x_train, y_train) in tqdm(enumerate(train_loader), ncols=50, desc="Training",
                                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        optimizer.zero_grad()
        x_train = x_train.to(device)
        output = net(x_train)
        loss = loss_fn(output, y_train.to(device))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        if batch_idx % 100 == 0:
            print(f"Step: {epoch * batch_idx + batch_idx}\t"
                  f"Epoch: {epoch} "
                  f"[{batch_idx * len(x_train)}/{len(train_loader.dataset)} "
                  f"({int(100 * batch_idx / float(len(train_loader)))}%)]\t"
                  f"Loss: {loss.item()}"
                  )

    # writer.add_scalar("on_epoch_loss", total_loss, epoch * batch_idx + batch_idx)


def test(train_test=False):
    correct = 0
    total = 0
    cm = np.zeros((10, 10), dtype=int)
    max_accuracy = 0
    with torch.no_grad():
        net.eval()
        total_loss = 0.
        for batch_idx, (x_test, y_test) in tqdm(enumerate(test_loader), ncols=50, desc="Testing",
                                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            output = net(x_test)
            test_loss = loss_fn(output, y_test)
            total_loss += test_loss.item()

            if batch_idx % 100 == 0:
                print(f"[{batch_idx * len(x_test)}/{len(test_loader.dataset)} "
                      f"({100 * batch_idx / float(len(test_loader))}%)]\t"
                      f"Loss: {test_loss.item()}"
                      )

            _, y_pred = torch.max(output.data, 1)
            total += y_test.size(0)
            correct += (y_pred == y_test).sum().item()
            compute_confusion_matrix(cm, y_test, y_pred)

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
        print(f"Confusion matrix:\n {cm}")

        plt.matshow(cm)
        plt.colorbar()
        plt.savefig(f"./images/cm_epoch{epoch}_accuracy{accuracy}.png")
        # plt.show()

        if train_test:
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                torch.save(net.state_dict(), f"./weights/weights_epoch{epoch}_accuracy{max_accuracy}.pth")

        # writer.add_scalar("test_accuracy", accuracy, epoch * batch_idx + batch_idx)
        # writer.add_scalar("on_epoch_test_loss", total_loss, epoch * batch_idx + batch_idx)


if __name__ == '__main__':
    for epoch in range(0, 100):
        train(epoch)
        test(train_test=True)
    test()
