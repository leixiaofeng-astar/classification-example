from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
from sklearn import metrics
import numpy as np
import pdb
import matplotlib.pyplot as plt
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.array(np.arange(N).reshape([-1, 1]) == ind, dtype=np.long)


def calc_loss(out, target):
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    loss = loss_fn(out, target)
    # weight = sample_weight.type('torch.FloatTensor').view([-1, 1])
    # loss = torch.sum(weight * torch.sum((out - target.type('torch.FloatTensor').cuda()) ** 4, 1).type('torch.FloatTensor'))
    # # loss = torch.sum(abs(out-target))
    # # loss = sum(loss * sample_weight)
    return loss


def train_model(model, data_loaders, optimizer, num_epochs=500, model_name='ResNet152'):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss_history_train = []
    epoch_loss_history_test = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            predict_result = []
            ground_truth = []
            Y_prob = []

#            pdb.set_trace()
            # Iterate over data.
            for imgs, labels in data_loaders[phase]:
                # imgs = imgs.to(device)
                # labels = labels.to(device)
                if torch.sum(imgs != imgs) > 1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    # Forward
                    features, out1 = model(imgs)
                    
                    loss = calc_loss(out1, labels)

                    #xiaofeng add for debug
                    #out = torch.softmax(self.fc(features), dim=1)
                    #cls_probs = torch.nn.functional.softmax(out1, dim=-1)
                    cls_probs = out1.data.cpu().numpy()
                    Y_prob.append(cls_probs[:,-1])

                    # predict_result.append(np.argmax(out.detach().cpu().numpy(), axis=1))
                    ground_truth.append(labels.cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # predict_result = np.hstack(predict_result)
            ground_truth = np.hstack(ground_truth)
            Y_prob = np.hstack(Y_prob)
            # import pdb
            # pdb.set_trace()
            # print("predict_result: ", predict_result)
            # print("Y_prob: ", Y_prob)
            if phase == 'train':
                epoch_loss_history_train.append(epoch_loss)
            else:
                epoch_loss_history_test.append(epoch_loss)

            # plt.scatter(ground_truth, predict_result, s=0.2)
            # plt.plot([0, 20], [0, 20], color='navy', lw=2, linestyle='--')
            # plt.xlabel('Ground truth')
            # plt.ylabel('Estimated')
            # plt.title('Regression plot')
            # save_file_name = "Res152_pre_new_model_" + str(epoch) + ".jpg"
            # plt.savefig(save_file_name, format='JPEG')
            # plt.close()
            # plt.show()

            # epoch_auc_score = metrics.roc_auc_score(ground_truth, predict_result)
            # print('{} {} Loss: {:.4f} AUC: {:.4f}'.format(model_name, phase, epoch_loss, epoch_auc_score))

            epoch_auc_score = metrics.roc_auc_score(ground_truth, Y_prob)
            print('{} {} Loss: {:.4f} AUC: {:.4f}'.format(model_name, phase, epoch_loss, epoch_auc_score))

            # deep copy the model
            if phase == 'val' and epoch_auc_score >= best_auc:
                best_auc = epoch_auc_score
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC score: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    epoch_loss_history = {'epoch_loss_history_train': epoch_loss_history_train,
                          'epoch_loss_history_test': epoch_loss_history_test}
    return model, epoch_loss_history
