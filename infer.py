import torch
import numpy as np
import sys
import cv2
from segmentation_models_pytorch import UnetPlusPlus
import albumentations as A

def mask_to_image(mask:np.ndarray):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    output[mask==1] = [255,0,0]
    output[mask==2] = [0,255,0]
    return output

def infer(img_path:str):
    model = UnetPlusPlus(
        classes=3
    )
    model.load_state_dict(torch.load("neopolyp_model_1.pth", weights_only=True))
    
    img = cv2.imread(img_path)
    img_size = (img.shape[0], img.shape[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    img = A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))(image=img)["image"]
    img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model.forward(img).squeeze(0).numpy().transpose(1,2,0)
    output = cv2.resize(output, img_size[::-1]).argmax(axis=2)
    mask_img = mask_to_image(output)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("mask.jpeg",mask_img)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        pass
    if sys.argv[1] == "--image_path":
        img_path = sys.argv[2]
        infer(img_path)
        print("Result saved as mask.jpeg")