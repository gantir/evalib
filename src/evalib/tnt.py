import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.plot_helper import plot_accuracy, plot_loss, plot_acc_loss

def l1_penalty(x):
    #L1 regularization adds an L1 penalty equal
    #to the absolute value of the magnitude of coefficients
    return torch.abs(x).sum()

def train_model(model, device, train_loader, optimizer, epoch, enable_l1_loss = False, l1_weight = 0.0001):
    model.train()
    train_loss = 0
    train_correct = 0
    train_acc = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if enable_l1_loss:
            to_reg = []
            for param in model.parameters():
                to_reg.append(param.view(-1))
            l1 = l1_weight*l1_penalty(torch.cat(to_reg))
        else:
            l1 = 0

        loss = F.nll_loss(output, target) + l1

        pred_train = output.argmax(dim=1, keepdim=True)
        train_correct += pred_train.eq(target.view_as(pred_train)).sum().item()

        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_acc = 100. * train_correct / len(train_loader.dataset)

        output_message = 'Epoch: {}, Batch {},  Training Loss: {:.8f} Training Accuracy: {:.4f}%'.format(epoch, batch_idx, train_loss,train_acc)
        pbar.set_description(desc= output_message)

    return train_acc, train_loss

def test_model(model, device, test_loader, epoch=1):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_acc = 100. * correct / len(test_loader.dataset)

    output_message = '\nEpoch: {} -- Test Set, Validation Loss: {:.8f} Validation Accuracy: {:.4f}%\n'.format(epoch,test_loss,test_acc)
    print(output_message)

    return test_acc, test_loss

def train_n_test(model, optimizer, scheduler, device, train_data_loader, test_data_loader, num_epochs=10, enable_l1_loss= False, l1_weight=0.0001):
  train_acc_history = []
  train_loss_history = []
  val_acc_history = []
  val_loss_history = []

  for epoch in range(1, num_epochs+1):
      train_acc, train_loss = train_model(model, device, train_data_loader, optimizer, epoch, enable_l1_loss, l1_weight)
      train_acc_history.append(train_acc)
      train_loss_history.append(train_loss)

      if scheduler is not None:
        print('LR:', scheduler.get_lr())
        scheduler.step()

      val_acc, val_loss = test_model(model, device, test_data_loader, epoch)
      val_acc_history.append(val_acc)
      val_loss_history.append(val_loss)

  return ([(train_acc_history, train_loss_history),(val_acc_history, val_loss_history)])
