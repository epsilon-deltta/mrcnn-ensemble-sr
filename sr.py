import cv2 as cv
from cv2 import dnn_superres
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import functional as vfc 
# bicubic, bilinear, nearest (,MetaSR)
class sr:
    
    def __init__(self,main='edsr',method='ensemble'):
        self.espcn = None
        self.fsrcnn = None
        self.edsr   = None
        self.model  = None
        self.method = method
        self.main   = main
            
        if main in ['edsr','espcn','fsrcnn','lapsrn']:
            self.sr = dnn_superres.DnnSuperResImpl_create()
            model_name = main
            fixedscale = 2
            if model_name == 'lapsrn':
                model_name = 'LapSRN'
#                 if scale2 == 3 : 
#                     raise ValueError("LapSRN has no 3x scale")
                path = f"./srmodels/{model_name}_x{fixedscale}.pb"
            else :
                path = f"./srmodels/{model_name.upper()}_x{fixedscale}.pb"

            self.sr.readModel(path)
            self.sr.setModel(model_name.lower(), fixedscale)
            
        elif main in ['rdn', 'esrgan', 'srgan'] :
            upimg = None
            
    def setmodel(self,main):
        pass
    
    def up2x(self,img):
        if self.main in ['edsr','espcn','fsrcnn','lapsrn']:
            upimg = self.sr.upsample(img)
            
        elif self.main in ['rdn', 'esrgan', 'srgan'] :
            upimg = None
            
        return upimg
            
    def np2torch(self,img,device): # numpy uint8 > torch float32
        return vfc.to_tensor(img).to(device='cuda:0')
        
    def torch2np(self,img):
        if 'cuda' in img.device.type:
            img = img.to('cpu')
        if img.dtype == torch.float32:
            img  = vfc.to_pil_image(img) # torch.uint8
        return np.array(img)
    
    def upsample(self,img,scale): #,main=None,method='ensemble'):
        # 
        back2torch = False
        
        if type(img) == torch.Tensor:
            if scale < 2 :
                # img = cv.resize(img, (uw,uh),interpolation=cv.INTER_CUBIC) 
                img = torch.nn.functional.interpolate(
                img[None], scale_factor=scale, mode='bilinear', recompute_scale_factor=True,
                align_corners=False)[0]
                return img

            device = img.device
            img = self.torch2np(img)
            back2torch = True
            
        
        methods = self.method.split('_')
        main    = self.main
        h,w = img.shape[:2]
        
        if methods[0].lower() == 'ensemble':
            
            main = main.lower()
            # inter + 2x_sr            
            uw,uh = int(w*scale),int(h*scale)
            fw,fh = uw,uh       

            if uw%2!=0:
                uw = uw -1 
            if uh%2!=0:
                uh = uh -1 
            uw = uw//2
            uh = uh//2        

            # inter 
            img = cv.resize(img, (uw,uh),interpolation=cv.INTER_CUBIC) 
            # 2x_sr 
            img = self.up2x(img)
            scale = 2
            
            if (uw != fw) or (uh != fh) :
                img = cv.resize(img, (fw,fh),interpolation=cv.INTER_CUBIC) 
        else:
            ValueError(f"There is no {methods[0]}")
            
        # img = np.transpose(img,(1,2,0))
        # img = np.transpose(img,(2,0,1))
        if back2torch:
            img = self.np2torch(img,device)
        
        return img