import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors
from utility.model_loading import load_model

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualise_predicted(model, image, name, path):
    img_ndarray = np.array(image)
    img_ndarray = img_ndarray[np.newaxis, ...] if img_ndarray.ndim == 2 else img_ndarray.transpose((2, 0, 1))
    img_ndarray = img_ndarray / 255
    image_tensor = torch.from_numpy(img_ndarray).unsqueeze(0).float()
    image_tensor = image_tensor.to(device=device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0] 
        probs = probs.detach().cpu().numpy().transpose((1, 2, 0))
        mask = np.argmax(probs, axis=2)
        Image.fromarray((mask).astype(np.uint8)).save(f"{path}/mask-{name}.png")
        
        color = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow', 'blue']) if len(np.unique(mask)) ==5 else colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow'])
        # color = colors.LinearSegmentedColormap.from_list("", ['white', 'red']) 
        plt.imshow(np.array(image))
        plt.imshow(mask, cmap=color, alpha=0.6)
        plt.axis('off')
        plt.savefig(f"{path}/{name}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
    return mask


def run_single():
    
    target = '8_ori_gf_10_proposed'
    model = load_model(channel=3, classes=5, bilinear=False, graph=True, graph_message_passing=True,\
                       model_path=f'checkpoint/{target}/loss=0.4108_dice=0.3924.pth').to(device=device)
    
    path = f"predicted/{target}"
    for i in ['f_01340', 'f_01522']:
        img = Image.open(f'data/spray/test/{i}.png').convert('RGB')
        mask = visualise_predicted(model, img, i, path)

run_single()