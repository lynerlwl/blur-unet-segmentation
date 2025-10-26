import torch
from pathlib import Path
from matplotlib import pyplot as plt

from utility.model_utils import load_model, load_image_to_tensor, get_model_outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name = "5490"
image_path = "data/wood/test/img/5490.jpg"
image_tensor = load_image_to_tensor(image_path)

'''
feature_map
'''

''' proposed '''
pathB = f"output/conv_feature_map/wood-proposedB/{name}"
Path(pathB).mkdir(parents=True, exist_ok=True)

checkpointB = "checkpoint/wood/ori_gf_proposed_loss=0.4616_dice=0.2441.pth"

modelB = load_model(channel=3, classes=2, graph=True, graph_message_passing=True, model_path=checkpointB).to(device=device)

outputsB = get_model_outputs(use_graph=True, model=modelB, image_tensor=image_tensor)

for num_block in range(len(outputsB)):
    xB = outputsB[num_block].squeeze(0)
    yB = torch.sum(xB,0)
    
    plt.imshow(yB.detach().cpu().numpy(), cmap='gray')# gray | hot
    plt.axis('off')
    plt.savefig(f"{pathB}/{num_block + 1}.png", pad_inches = 0, bbox_inches='tight', dpi=300);
