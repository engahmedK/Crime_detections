from ast import arg
import numpy as np
import os
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
import cv2
import utils

from basicsr.models import create_model
from basicsr.utils.options import parse

def enhance_video(input_path, result_dir='./results/', weights='pretrained_weights/SDSD_indoor.pth', gpus='0'):
    # Set GPU
    gpu_list = ','.join(str(x) for x in gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # Load model and weights
    opt = parse('Options/RetinexFormer_SDSD_indoor.yml', is_train=False)
    opt['dist'] = False
    model_restoration = create_model(opt).net_g

    checkpoint = torch.load(weights)
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    print("===> Testing using weights: ", weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the enhanced video
    output_path = os.path.join(result_dir, 'enhanced_video.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process each frame in the video
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to float32 and normalize
            img = np.float32(frame) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding
            factor = 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            input_ = F.pad(input_, (0, W - w, 0, H - h), 'reflect')

            # Enhancement
            restored = model_restoration(input_)

            # Unpad
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # Convert back to uint8
            restored_uint8 = img_as_ubyte(restored)

            # Write the enhanced frame to the output video
            out.write(restored_uint8)

        # Release VideoCapture and VideoWriter objects
        cap.release()
        out.release()

    print(f"Enhanced video saved at: {output_path}")

# Example usage:
# enhance_video(input_path='./path/to/your/video.mp4', result_dir='./results/', weights='pretrained_weights/SDSD_indoor.pth', gpus='0')
