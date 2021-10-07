import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import argparse

parser = argparse.ArgumentParser("Train code")   
# setting
parser.add_argument('-m'  ,'--mode'   ,default=-1, type =int  ,metavar='{...}'    ,help="mode index 0: 'edsr','espcn','fsrcnn','lapsrn','bilinear','bicubic'] ")
parser.add_argument('-b'  ,'--batch'   ,default=-1, type =int  ,metavar='{...}'    ,help="batch size ")
args = parser.parse_args()

def get_idle_gpu():
    import GPUtil as gp
    gpus = gp.getGPUs()
    gpus_util = [gpu.memoryUtil for gpu in gpus]
    
    import numpy as np
    index = np.argmin(gpus_util)
    device = 'cuda:'+str(index)
    return device

saved  = False
device = get_idle_gpu()
device = 'cuda:0'
num_epochs = 20
batch_size = args.batch if args.batch != -1 else 4
pretrained = False
mode_names = ['edsr','espcn','fsrcnn','lapsrn','bilinear','bicubic']
mode_index = args.mode if args.mode != -1 else 1
# mode_index = 1 
mode_name = 'ensemble-'+ mode_names[mode_index] if mode_index in range(0,4) else mode_names[mode_index]
classes = ['Nothing','Pedestrian']    

print(f'{mode_name} is started on the {device}')
print(f'batch {batch_size}')

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"]    = boxes
        target["labels"]   = labels
        target["masks"]    = masks
        target["image_id"] = image_id
        target["area"]     = area
        target["iscrowd"]  = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from rcnn_transfrom import InterpolationTransform as it
      
def get_instance_segmentation_model(num_classes,pretrained=True,mode_name='ensemble-edsr'):
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
    
    model.transform = it(min_size=(800,), max_size=1333,image_mean=[0.485, 0.456, 0.406],image_std=[0.229, 0.224, 0.225],mode=mode_name)
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

dataset      = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


num_classes = 2


model = get_instance_segmentation_model(num_classes,pretrained=pretrained,mode_name=mode_name)


model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


from tqdm import notebook as nb
import tqdm
evaluators = []
for epoch in range(num_epochs):
    

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    lr_scheduler.step()
    # evaluate on the test dataset
    # device = 'cuda:1'
    # model.to(device)
    evaluators.append( evaluate(model, data_loader_test, device=device) )

if saved:
    if pretrained :
        torch.save({'state_dict':model.state_dict(),
            'evaluators':evaluators
            },f'./model/pf_{batch_size}_{mode_name}.pth')
    else:
        torch.save({'state_dict':model.state_dict(),
            'evaluators':evaluators
            },f'./model/pf_{batch_size}_{mode_name}_{pretrained}.pth')
else:
    pass