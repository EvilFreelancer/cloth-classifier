import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from cloth_classifier.prepare_dataset import prepare_dataset
from cloth_classifier.models.rhombus_net import RhombusNet
from cloth_classifier.models.simple_net import SimpleNet
from cloth_classifier.models.advanced_net import AdvancedNet


def train(model, loss_fn, optimizer, trainloader, testloader, device, n_epochs=10):
    train_losses = []
    test_losses = []
    best_epoch_acc = 0
    for epoch in range(n_epochs):
        # Set mode to training - Dropouts will be used here
        model.train()
        train_epoch_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # flatten the images to batch_size x 784
            images = images.view(images.shape[0], -1)
            # forward pass
            outputs = model(images)
            # backpropogation
            train_batch_loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            train_batch_loss.backward()
            # Weight updates
            optimizer.step()
            train_epoch_loss += train_batch_loss.item()
        else:
            # One epoch of training complete
            # calculate average training epoch loss
            train_epoch_loss = train_epoch_loss / len(trainloader)

            # Now Validate on testset
            with torch.no_grad():
                test_epoch_acc = 0
                test_epoch_loss = 0
                # Set mode to eval - Dropouts will NOT be used here
                model.eval()
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    # flatten images to batch_size x 784
                    images = images.view(images.shape[0], -1)
                    # make predictions
                    test_outputs = model(images)
                    # calculate test loss
                    test_batch_loss = loss_fn(test_outputs, labels)
                    test_epoch_loss += test_batch_loss

                    # get probabilities, extract the class associated with highest probability
                    proba = torch.exp(test_outputs)
                    _, pred_labels = proba.topk(1, dim=1)

                    # compare actual labels and predicted labels
                    result = pred_labels == labels.view(pred_labels.shape)
                    batch_acc = torch.mean(result.type(torch.FloatTensor))
                    test_epoch_acc += batch_acc.item()
                else:
                    # One epoch of training and validation done
                    # calculate average testing epoch loss
                    test_epoch_loss = test_epoch_loss / len(testloader)
                    # calculate accuracy as correct_pred/total_samples
                    test_epoch_acc = test_epoch_acc / len(testloader)

                    if torch.is_tensor(train_epoch_loss):
                        train_epoch_loss = train_epoch_loss.cpu().numpy()
                    if torch.is_tensor(test_epoch_loss):
                        test_epoch_loss = test_epoch_loss.cpu().numpy()

                    # save epoch losses for plotting
                    train_losses.append(train_epoch_loss)
                    test_losses.append(test_epoch_loss)

                    # print stats for this epoch
                    print(f'Epoch: {epoch} -> train_loss: {train_epoch_loss:.19f}, val_loss: {test_epoch_loss:.19f},',
                          f'val_acc: {test_epoch_acc * 100:.2f}%')

                    if test_epoch_acc > best_epoch_acc:
                        print(f'saving model...')
                        best_epoch_acc = test_epoch_acc
                        torch.save(model.state_dict(), './models/cloth_model.pth')

    # Finally plot losses
    plt.plot(train_losses, label='train-loss')
    plt.plot(test_losses, label='val-loss')
    plt.legend()
    plt.savefig('train_loss.png')


if __name__ == '__main__':
    # Prepare dataset
    train_dl, val_dl, test_dl = prepare_dataset(batch_size=64, img_height=64, img_weight=64)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init model
    # model = SimpleNet(input_size=4096).to(device)
    # model = AdvancedNet(input_size=4096).to(device)
    model = RhombusNet(input_size=4096).to(device)

    # Define the criterion and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)

    # Start the train loop
    train(model, loss_fn, optimizer, train_dl, val_dl, device, n_epochs=50)
