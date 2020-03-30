import torch
from tqdm import tqdm

from evalib import utils


def l1_penalty(x):
    # L1 regularization adds an L1 penalty equal
    # to the absolute value of the magnitude of coefficients
    return torch.abs(x).sum()


def train(model, train_loader, criterion, optimizer, l1_decay=1e-3):
    device = utils.get_device()
    model.train()

    train_loss = 0
    train_correct = 0
    train_acc = 0

    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do
        # backpropragation because PyTorch accumulates the gradients on subsequent
        # backward passes.

        # Because of this, when you start your training loop, ideally you should
        # zero out the gradients so that you do the parameter update correctly.

        output = model(data)

        # Calculate loss
        loss = criterion(output, target)
        l1_loss = 0
        if 0 < l1_decay:
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)

            l1_loss = l1_decay * l1_loss
        loss += l1_loss

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_acc = 100.0 * train_correct / len(train_loader.dataset)

        output_message = """Batch {},
            Training Loss: {:.8f}, Training Accuracy: {:.4f}%""".format(
            batch_idx, train_loss, train_acc
        )
        pbar.set_description(desc=output_message)

    return train_acc, train_loss


def test(model, test_loader, criterion):
    device = utils.get_device()
    model.eval()
    test_loss = 0
    correct = 0

    test_acc = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    output_message = """\nTest Set,
        Test Loss: {:.8f}, Test Accuracy: {:.4f}%\n""".format(
        test_loss, test_acc
    )
    print(output_message)

    return test_acc, test_loss


def train_n_test(
    model,
    criterion,
    optimizer,
    scheduler,
    train_data_loader,
    test_data_loader,
    num_epochs=10,
    l1_decay=1e-3,
):
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train(
            model, train_data_loader, criterion, optimizer, l1_decay
        )
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        if scheduler is not None:
            print("LR:", scheduler.get_lr())
            scheduler.step()

        val_acc, val_loss = test(model, test_data_loader, criterion)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

    return [
        (train_acc_history, train_loss_history),
        (val_acc_history, val_loss_history),
    ]
