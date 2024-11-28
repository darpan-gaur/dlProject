#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from scipy import ndimage
from scipy.ndimage.interpolation import zoom

# import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.optim as optim
from utils import DiceLoss
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# In[2]:


# set seed
seed = 304
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[3]:


GLOB_PATH = '/export/home/darpan/work_dir/aps/dl/data'

# Path to the director where images for semantic segmentation are stored
IMAGES_DIR = f'{GLOB_PATH}/Image/'
# Path to the directory where labels for semantic segmentation are stored
LABELS_DIR = f'{GLOB_PATH}/Mask/'
IMG_EXT = 'jpg'
LABEL_EXT = 'png'


# In[4]:


metadata = pd.read_csv(f'{GLOB_PATH}/metadata.csv')
metadata.head()


# In[5]:


def get_all_images_labels(IMAGES_DIR, LABELS_DIR, metadata):
    """
    Return the list of all valid images and labels
    """
    images = []
    labels = []
    min_h = 1e6; max_h = 0
    min_w = 1e6; max_w = 0
    for _, row in metadata.iterrows():
        image = os.path.join(IMAGES_DIR, row['Image'])
        label = os.path.join(LABELS_DIR, row['Mask'])
        img_arr = np.array(Image.open(image))
        label_arr = np.array(Image.open(label))
        if (img_arr.ndim == 3) and (img_arr.shape[2] == 3) and (img_arr.shape[:-1] == label_arr.shape):
            min_h = min(min_h, img_arr.shape[0]); max_h = max(max_h, img_arr.shape[0])
            min_w = min(min_w, img_arr.shape[1]); max_w = max(max_w, img_arr.shape[1])
            images.append(image)
            labels.append(label)
        else:
            print(f"Skipping image {row['Image']} and mask {row['Mask']}")
    print(f"Min height: {min_h}, Max height: {max_h}")
    print(f"Min width: {min_w}, Max width: {max_w}")
    return images, labels


# In[6]:


# Take the first TRAIN_SIZE images for training
all_images, all_labels = get_all_images_labels(IMAGES_DIR, LABELS_DIR, metadata)

print(len(all_images), len(all_labels))
print(all_images[:5])
print(all_labels[:5])


# In[7]:


TRAIN_SIZE = 200 # Number of images to use for training
VAL_SIZE = 20 # Number of images to use for validation
TEST_SIZE = 60 # Number of images to use for testing


# In[8]:


image_paths_train = all_images[:TRAIN_SIZE]
label_paths_train = all_labels[:TRAIN_SIZE]

print(image_paths_train[:5])
print(label_paths_train[:5])

image_paths_val = all_images[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
label_paths_val = all_labels[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]

print(image_paths_val[:5])
print(label_paths_val[:5])

image_paths_test = all_images[-TEST_SIZE:]
label_paths_test = all_labels[-TEST_SIZE:]

print(image_paths_test[:5])
print(label_paths_test[:5])


# **Note:** White color in mask means flooded area and black color means non-flooded area.

# ## Dataloader

# In[9]:


transform = transforms.Compose([
    # resize the image to (256, 256)
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# In[10]:


class floodDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        # convert label to binary
        label = (label > 0).float()
        return image, label


# In[11]:


BATCH_SIZE = 4


# In[12]:


db_train = floodDataset(image_paths_train, label_paths_train, transform=transform)
db_test = floodDataset(image_paths_test, label_paths_test, transform=transform)


# In[13]:


trainloader = DataLoader(db_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(db_test, batch_size=BATCH_SIZE, shuffle=False)


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# In[15]:


MODEL_NAME = 'R50-ViT-B_16'
config_vit = CONFIGS_ViT_seg[MODEL_NAME]
config_vit.n_classes = 2
model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).to(device)
model.load_from(weights=np.load('/export/home/darpan/work_dir/aps/dl/TransUNet/model/R50+ViT-B_16.npz'))


# In[16]:


model.train()
LR = 0.01
num_classes = 2
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(num_classes)
MOMENTUM = 0.9
DECAY = 1e-4
opt_name = 'sgd'
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
MAX_EPOCHS = 150

# iter


# In[17]:


res_dir = './results'
exp_name = f'{MODEL_NAME}_3_1_loss_ratio'
os.makedirs(f'{res_dir}/{exp_name}', exist_ok=True)

config = {
    'model': MODEL_NAME,
    'lr': LR,
    'epochs': MAX_EPOCHS,
    'batch_size': BATCH_SIZE,
    'optimizer': opt_name,
    'momentum': MOMENTUM,
    'decay': DECAY,
    'seed': seed
}

with open(f'{res_dir}/{exp_name}/config.json', 'w') as f:
    json.dump(config, f)


# In[18]:


def eval_model(model, testloader, optimizer, log_file, work='test'):
    model.eval()
    iou_score = 0
    dice_score = 0
    loss_t = 0
    loss_ce_t = 0
    loss_dice_t = 0
    for i, data in enumerate(testloader):
        image, label = data

        image, label = image.to(device), label.to(device)
        outputs = model(image)

        label = label.squeeze(1)
        label2 = torch.nn.functional.one_hot(label.long(), num_classes=2).permute(0, 3, 1, 2).float()
        # print(outputs.shape, label2.shape)
        # print(outputs.long().dtype, label.dtype)
        loss_ce = ce_loss(outputs, label2)
        loss_dice = dice_loss(outputs, label)
        loss = 0.75 * loss_ce + 0.25 * loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # calulate iou score and dice score
        outputs = torch.argmax(outputs, dim=1)
        label = label.squeeze(1)
        

        tp = torch.sum(outputs * label)
        fp = torch.sum(outputs * (1 - label))
        fn = torch.sum((1 - outputs) * label)
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)

        
        iou_score += iou
        dice_score += dice
        loss_t += loss
        loss_ce_t += loss_ce
        loss_dice_t += loss_dice
        # break
    iou_score /= len(testloader)
    dice_score /= len(testloader)
    loss_t /= len(testloader)
    loss_ce_t /= len(testloader)
    loss_dice_t /= len(testloader)
    print(f'{work} Data: IoU: {iou_score}, Dice: {dice_score}, Loss: {loss_t}, CE Loss: {loss_ce_t}, Dice Loss: {loss_dice_t}')
    log_file.write(f'{work} Data: IoU: {iou_score}, Dice: {dice_score}, Loss: {loss_t}, CE Loss: {loss_ce_t}, Dice Loss: {loss_dice_t}\n')

    return iou_score, dice_score, [loss_t, loss_ce_t, loss_dice_t]


# In[ ]:


def train_model(model, trainloader, testloader, valloader, optimizer, log_file, max_epoch=100):
    
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = 0
    iou_val_best = 0
    iou_test_best = 0
    for epoch_num in iterator:
        model.train()
        iou_score = 0
        dice_score = 0
        loss_t = 0
        loss_ce_t = 0
        loss_dice_t = 0
        for i, data in enumerate(trainloader):
            image, label = data

            image, label = image.to(device), label.to(device)
            outputs = model(image)

            label = label.squeeze(1)
            label2 = torch.nn.functional.one_hot(label.long(), num_classes=2).permute(0, 3, 1, 2).float()
            loss_ce = ce_loss(outputs, label2)
            loss_dice = dice_loss(outputs, label)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            outputs = torch.argmax(outputs, dim=1)
            label = label.squeeze(1)
            tp = torch.sum(outputs * label)
            fp = torch.sum(outputs * (1 - label))
            fn = torch.sum((1 - outputs) * label)
            iou = tp / (tp + fp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)

            # print(f"Epoch: {epoch_num}, Iteration: {i}, Loss: {loss}, CE Loss: {loss_ce}, Dice Loss: {loss_dice}, IoU: {iou}, Dice: {dice}")
            log_file.write(f"Epoch: {epoch_num}, Iteration: {i}, Loss: {loss}, CE Loss: {loss_ce}, Dice Loss: {loss_dice}, IoU: {iou}, Dice: {dice}\n")
            iou_score += iou
            dice_score += dice
            loss_t += loss
            loss_ce_t += loss_ce
            loss_dice_t += loss_dice
        iou_score /= len(trainloader)
        dice_score /= len(trainloader)
        loss_t /= len(trainloader)
        loss_ce_t /= len(trainloader)
        loss_dice_t /= len(trainloader)
        print(f"Epoch: {epoch_num}, Loss: {loss_t}, CE Loss: {loss_ce_t}, Dice Loss: {loss_dice_t}, IoU: {iou_score}, Dice: {dice_score}")
        log_file.write(f"Epoch: {epoch_num}, Loss: {loss_t}, CE Loss: {loss_ce_t}, Dice Loss: {loss_dice_t}, IoU: {iou_score}, Dice: {dice_score}\n")

        iou_val, dice_val, loss_val = eval_model(model, valloader, optimizer, log_file, work='val')
        iou_test, dice_test, loss_test = eval_model(model, testloader, optimizer, log_file, work='test')

        if (iou_val > iou_val_best):
            iou_val_best = iou_val
            torch.save(model.state_dict(), f'{res_dir}/{exp_name}/best_model_val.pth')
            log_file.write(f"Best model val saved at epoch {epoch_num} with IoU: {iou_val_best}\n")

        if (iou_test > iou_test_best):
            iou_test_best = iou_test
            torch.save(model.state_dict(), f'{res_dir}/{exp_name}/best_model_test.pth')
            log_file.write(f"Best model test saved at epoch {epoch_num} with IoU: {iou_test_best}\n")
    


# In[ ]:


log_file = open(f'{res_dir}/{exp_name}/log.txt', 'w')

train_model(model, trainloader, testloader, testloader, optimizer, log_file, max_epoch=MAX_EPOCHS)

torch.save(model.state_dict(), f'{res_dir}/{exp_name}/final_model.pth')

iou_test, dice_test, _ = eval_model(model, testloader, optimizer, log_file, work='test')

log_file.close()


# In[ ]:




