import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
import torchvision.transforms as transforms
from multi_read_data import MemoryFriendlyLoader
import cv2

def load_image(frame):
    
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image= Image.fromarray(image)
    return image

def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    img = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def video(input,out_put):

    video = cv2.VideoCapture(input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_put, fourcc, 20.0, (width,height))
    model = Finetunemodel('./weights/difficult.pt')
    model = model.cuda()
    transform = [
            transforms.ToTensor(),
        ]
    transformed = transforms.Compose(transform)
    with torch.no_grad():
    # Load the single image

        while True:
            ret, frame = video.read()
            if not ret:
                break
            print(frame.shape)
            input = load_image(frame)  
            input = transformed(input).unsqueeze(0)
        # Preprocess the input
            input = Variable(input, volatile=True).cuda()
            i, r = model(input)
            output = save_images(r)
            out.write(output)
            # Process the input through the model
        
    
