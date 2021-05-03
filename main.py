'''
    Instructions to use:
        python3 main.py
        python3.7 main.py

    description:
        2-class classification task on chest Xay images

    What to do:
        the Xray images is under datadir folder in the code
        default directory is 'chest_xray'
        you need to change it to your own data folder with the same data structure
        There are 3 folder for train/val/test data set in
        traindir = datadir + '/train/'
        validdir = datadir + '/val/'
        testdir = datadir + '/test/'

        You can change models_list in main function to use different backbone to train the model

    original xray images come from
    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download
    Note: the val dataset dividing is poor, it has only 16 images, you can change it

'''

from torchvision import transforms, datasets
from torch import cuda
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import metrics
from train_modal import train_model
from own_models import *
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import json
import time
import argparse
import pickle


image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return v


def get_model(model_name, num_class):
    # 'VGG16', 'VGG19bn', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    #                    'DenseNet121', 'DenseNet169', 'DenseNet201'
    if model_name == 'VGG16':
        model = VGG16(num_class)
        return model
    if model_name == 'VGG19bn':
        model = VGG19bn(num_class)
        return model
    if model_name == 'ResNet18':
        model = ResNet18(num_class)
        return model
    if model_name == 'ResNet34':
        model = ResNet34(num_class)
        return model
    if model_name == 'ResNet50':
        model = ResNet50(num_class)
        return model
    if model_name == 'ResNet101':
        model = ResNet101(num_class)
        return model
    if model_name == 'ResNet152':
        model = ResNet152(num_class)
        return model
    if model_name == 'DenseNet121':
        model = DenseNet121(num_class)
        return model
    if model_name == 'DenseNet169':
        model = DenseNet169(num_class)
    if model_name == 'DenseNet201':
        model = DenseNet201(num_class)
        return model
    if model_name == 'EfficientNet_b7':
        model = EfficientNetb7(num_class=num_class)
        return model


def parse_args():
    # generated using arg gen version:0.2_lite
    CLI = argparse.ArgumentParser()
    # input params
    CLI.add_argument(
        "--zip_path", type=str, default="chest_xray",
    )
    # output params
    CLI.add_argument(
        "--output", type=str, default="prediction.txt",
    )
    CLI.add_argument(
        "--output_model", type=str, default="best_model4ResNet18",
    )
    return CLI.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    datadir = args.zip_path
    output = args.output
    output_model = args.output_model

    # Loading data
    # datadir = 'chest_xray'
    # datadir = '/data2/home/xiaofeng/covid_chestxray/chest_xray'
    print("Data Location: ", os.listdir(datadir))
    traindir = datadir + '/train/'
    validdir = datadir + '/val/'
    testdir = datadir + '/test/'

    # Change to fit hardware
    batch_size = 16 # 256
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print('Train on gpu:{}'.format(train_on_gpu))

    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()

        print('{} gpus detected.'.format(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    # Empty lists
    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # Iterate through each category
    for d in os.listdir(traindir):
        if not d.startswith('.'):
            categories.append(d)

            # Number of each image
            train_imgs = os.listdir(traindir + d)
            valid_imgs = os.listdir(validdir + d)
            test_imgs = os.listdir(testdir + d)
            n_train.append(len(train_imgs))
            n_valid.append(len(valid_imgs))
            n_test.append(len(test_imgs))

            # Find stats for train images
            for i in train_imgs:
                if not i.startswith('.'):
                    img_categories.append(d)
                    img = Image.open(traindir + d + '/' + i)
                    img_array = np.array(img)
                    # Shape
                    hs.append(img_array.shape[0])
                    ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid, 'n_test': n_test}).\
        sort_values('category')

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })


    # for debug usage: Show two examples of normal and pneumonia cases.
    # x = Image.open(traindir + 'NORMAL/IM-0128-0001.jpeg')
    # np.array(x).shape
    # imshow(x)
    #
    # x = Image.open(traindir + 'PNEUMONIA/person1001_bacteria_2932.jpeg')
    # np.array(x).shape
    # imshow(x)

    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
        datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    len(data['train'].classes)
    print(data['train'].class_to_idx)
    print(' ***** Data loading is completed ***** ')


    # models_list = ['VGG16', 'VGG19bn', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    #                'DenseNet121', 'DenseNet169', 'DenseNet201']

    # Select your model#
    # models_list = ['ResNet18', 'ResNet50', 'DenseNet121']
    models_list = ['EfficientNet_b7']
    MAX_EPOCH = 16
    lr = 1e-4
    betas = (0.5, 0.999)

    performance_list = []
    for model_name in models_list:
        # folder_name = model_name + "_" + "betaC2"
        folder_name = './'
        model_ft = get_model(model_name, 2)
        print(' ***** Load {} successfully ***** '.format(model_name))

        if train_on_gpu:
            model_ft = model_ft.to('cuda')
        if multi_gpu:
            model_ft = nn.DataParallel(model_ft)

        params_to_update = list(model_ft.parameters())

        # Observe that all parameters are being optimized
        optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
        # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        print(' ***** Training is beginning ***** ')
        # Train and evaluate
        model_ft, loss_hist = train_model(model_ft, dataloaders, optimizer, MAX_EPOCH, model_name)
        print(' ***** Training is completed ***** ')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # my_model_path = folder_name + '/best_model4' + model_name
        my_model_path = os.path.join(folder_name, output_model)
        torch.save(model_ft, my_model_path)

        # Testing
        #model_ft2 = get_model(model_name, 2)
        del model_ft
        time.sleep(2)
        model_ft2 = torch.load(my_model_path)
        model_ft2.eval()

        '''
        # Save/Load Entire Model
        torch.save(model, PATH)
        model = torch.load(PATH)
        model.eval()
        
        # Saving & Loading Model for Inference
        torch.save(model.state_dict(), PATH)
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        '''

        predict_test = []
        label_test = []

        for imgs, labels in dataloaders['test']:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                model_ft2 = model_ft2.to('cuda')

            if model_name == 'EfficientNet_b7':
                features, out1 = model_ft2(imgs)
            else:
                features, out, out1 = model_ft2(imgs)
            predict_test.append(out1.detach().cpu().numpy()[:, -1])
            label_test.append(labels.numpy())
        predict_test = np.concatenate(predict_test)
        label_test = np.concatenate(label_test)
        test_auc_score = metrics.roc_auc_score(label_test, predict_test)
        test_precision_score = metrics.precision_score(label_test, predict_test > 0.5)
        results_report = metrics.confusion_matrix(label_test, predict_test > 0.5)

        print('Test confusion_matrix:')
        print(results_report)

        print(' ***** {} Best AUC score: {:.4f} on test set***** '.format(model_name, test_auc_score))
        print(' ***** {} Best test_precision_score: {:.4f} on test set***** '.format(model_name, test_precision_score))
        performance_list.append({'model_name': model_name, 'test_auc_score': test_auc_score,
                                 'test_precision_score': test_precision_score})
    print(performance_list)


    with open(output, 'w') as file_handle:
        json.dump(performance_list, file_handle)
