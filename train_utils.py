import torch
import numpy as np
import torch.autograd as ta

def train_with_momentum(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    """
    Trains a model for one epoch using an optimizer that supports per-sample momentum and low-pass filter.
    
    This function handles the training loop for a single epoch, including:
    - Forward and backward passes
    - Gradient accumulation over multiple batches if needed
    - Optimizer stepping with momentum-based optimizers
    - Accuracy and loss calculation
    - Logging of training metrics
    
    Args:
        model: The neural network model to train
        train_dl: Training data loader
        optimizer: Optimizer with optional momentum support
        criterion: Loss function
        log_file: Logger for recording metrics
        device: Device to run training on (default: 'cpu')
        epoch: Current epoch number (default: -1)
        log_frequency: How often to log metrics (default: -1, meaning no logging)
        acc_step: Number of batches to accumulate gradients over (default: 1)
        lr_scheduler: Optional learning rate scheduler (default: None)
    """
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for t, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)
        def closure(scale = 1.0):
            predict = model(input)
            if not isinstance(predict, torch.Tensor):
                predict = predict.logits
            loss = criterion(predict, label)
            scaled_loss = loss*scale
            scaled_loss.backward()
            return loss, predict
        
        if hasattr(optimizer, 'prestep'):
            loss, predict = optimizer.prestep(closure)
        else:
            loss, predict = closure()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        del input
        del label
        del loss
        del predict

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])

@torch.no_grad()
def test(model, test_dl, criterion, log_file, device = 'cpu', epoch = -1, **kwargs):
    """
    Evaluates a model on the test dataset.
    
    This function:
    - Sets the model to evaluation mode
    - Computes loss and accuracy metrics on the test dataset
    - Logs the results
    - Does not update model parameters
    
    Args:
        model: The neural network model to evaluate
        test_dl: Test data loader
        criterion: Loss function
        log_file: Logger for recording metrics
        device: Device to run evaluation on (default: 'cpu')
        epoch: Current epoch number (default: -1)
        **kwargs: Additional keyword arguments
    """
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(" ")
    log_file.update([epoch, -1],[100.*correct/total, test_loss/(batch_idx+1)])
