import argparse
import torch
import torch.nn as nn
from network.Math_Module import P, Q
from network.decom import Decom
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
from utils import *
import cv2




def np_save_TensorImg2(img_tensor):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    img = np.clip(img * 255, 0, 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img 

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class Inference(nn.Module):
    def __init__(self):
        super().__init__()
        
        # loading decomposition model
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, "./ckpt/init_low.pth")
        self.model_Decom_low = self.model_Decom_low.cuda()  # Move to GPU
        # loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding("./ckpt/unfolding.pth")
        self.model_R = self.model_R.cuda()  # Move to GPU
        self.model_L = self.model_L.cuda()  # Move to GPU
        # loading adjustment model
        self.adjust_model = load_adjustment("./ckpt/L_adjust.pth")
        self.adjust_model = self.adjust_model.cuda()  # Move to GPU
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)#time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0: # initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else: # update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def lllumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * 5
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start = time.time()  
            R, L = self.unfolding(input_low_img)
            High_L = self.lllumination_adjust(L, 5)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def run(self, low_img):

        
        
        img = cv2.cvtColor(low_img,cv2.COLOR_BGR2RGB)

        img= Image.fromarray(img)
        low_img = self.transform(img).unsqueeze(0)
        enhance, _ = self.forward(input_low_img=low_img)
        return np_save_TensorImg2(enhance) 
         
        

if __name__ == "__main__":
    def video(input,output):
        current_directory = os.getcwd()
        os.chdir(current_directory)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        model = Inference()
        ### in

        video = cv2.VideoCapture(input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        ### out 
        out = cv2.VideoWriter(output, fourcc, 20.0, (width,height))
        while True:
            ret, frame = video.read()
            if not ret:
                break
            print(frame.shape)
            output = model.run(frame)
             
            print(output.shape)
            cv2.imshow('img',output)
            cv2.waitKey(10)
            out.write(output)



    input = "D:\\WorkStation\\studing\\URetinex-Net\\video_crime\\Vandalism009_x264.mp4"
    output ="D:\\WorkStation\\studing\\URetinex-Net\\output_crime\\Vandalism009_x264.avi"
    video(input,output)
