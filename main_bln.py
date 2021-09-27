import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
from torchvision.transforms import functional as vtf
from PIL import ImageDraw 
import re

num_epochs = 20
batch_size = 4
mode_names = ['edsr','espcn','fsrcnn','lapsrn']
mode_name  = 'ensemble-'+ mode_names[1]

train_root = './balloon/balloon/train'
val_root   = './balloon/balloon/val'
train_ann  = './balloon/balloon/train/via_region_data.json'
val_ann    = './balloon/balloon/val/via_region_data.json'

class BalloonDataset(torch.utils.data.Dataset):
    def __init__(self, root='./balloon/balloon/train', annFile='./balloon/balloon/train/via_region_data.json',transforms=None):
        self.root = root
        self.transforms = transforms
        
        with open(annFile,'r') as f:
            self.ans = json.load(f)
        self.indices = sorted(self.ans.keys() )
        postfix = re.compile('.jpg\d+')
        self.files   = [re.sub(postfix,'.jpg',fname) for fname in self.indices]
        
        self.hw = []
        for fname in self.files:
            fpath = os.path.join(root,fname)
            fimg = Image.open(fpath).convert("RGB")
            h,w = fimg.height , fimg.width
            self.hw.append((h,w) )
            del fimg
            
    def __getitem__(self, idx):
        key = self.indices[idx]
        it  = self.ans[key]
        h,w = self.hw[idx]
        
        img_path = os.path.join(self.root,self.files[idx])
        img = Image.open(img_path).convert("RGB")
        num_objs = len(it['regions'])

        boxes = []
        masks = []
        for i in range(num_objs):
            x_ = it['regions'][str(i)]['shape_attributes']['all_points_x']
            y_ = it['regions'][str(i)]['shape_attributes']['all_points_y']
            # boxes
            xmin = np.min(x_)
            xmax = np.max(x_)
            ymin = np.min(y_)
            ymax = np.max(y_)
            boxes.append([xmin, ymin, xmax, ymax])
            
            # masks
            zimg = torch.zeros(h,w)
            pimg = vtf.to_pil_image(zimg)
            dr = ImageDraw.Draw(pimg)
            dr.polygon(list(zip(x_,y_)) ,fill=1)
            nimg = np.array(pimg)
            masks.append(nimg)
            del dr
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks,dtype=torch.uint8)
        area  = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])        
        
        target = {}
        
        
        target["masks"]    = masks
        target["boxes"]    = boxes
        
        target["area"]     = area
        
        target["image_id"] = image_id
        target["labels"]   = labels
        target["iscrowd"]  = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes,pretrained=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset      = BalloonDataset(train_root,train_ann, get_transform(train=True))
dataset_test = BalloonDataset(val_root,val_ann, get_transform(train=False))

torch.manual_seed(1)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cuda:1'
num_classes = 2


model = get_instance_segmentation_model(num_classes,pretrained=False)

####################################
from rcnn_transfrom import InterpolationTransform as it

model.transform = it(min_size=(800,), max_size=1333,image_mean=[0.485, 0.456, 0.406],image_std=[0.229, 0.224, 0.225],mode=mode_name)
#################################
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

from tqdm import notebook as nb

evaluators = []
for epoch in nb.tqdm(range(num_epochs)):
    
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    lr_scheduler.step()
    # evaluate on the test dataset
    # device = 'cuda:1'
    # model.to(device)
    evaluators.append( evaluate(model, data_loader_test, device=device) )



torch.save({'state_dict':model.state_dict(),
           'evaluators':evaluators
           },f'./model/bln_4_{mode_name}.pth')