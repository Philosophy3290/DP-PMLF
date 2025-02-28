import torch
import numpy as np
import torch.autograd as ta

def train_with_momentum(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
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

        # if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
        #     print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
        #     if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
        #         log_file.update([epoch, t],[100.*correct/total, train_loss])