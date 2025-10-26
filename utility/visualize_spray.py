import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from utility.model_utils import load_model, load_image_to_tensor, get_model_outputs
from model.graph_utils import proposed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
orig_map=plt.get_cmap('gist_heat')  
reversed_map = orig_map.reversed() 

name = "f_01340"
image_path = "data/spray/test/f_01340.png"
image_tensor = load_image_to_tensor(image_path)

'''
feature_map
'''

''' euclidean '''
pathA = f"output/conv_feature_map/spray-euclideanA/{name}"
Path(pathA).mkdir(parents=True, exist_ok=True)
pathB = f"output/conv_feature_map/spray-euclideanB/{name}"
Path(pathB).mkdir(parents=True, exist_ok=True)

checkpointA = "checkpoint/spray/ori_euclidean-loss=0.5437_dice=0.3910.pth"
checkpointB = "checkpoint/spray/ori_gf_euclidean-loss=0.5985_dice=0.4023.pth"

''' proposed '''
pathA = f"output/conv_feature_map/spray-proposedA/{name}"
Path(pathA).mkdir(parents=True, exist_ok=True)
pathB = f"output/conv_feature_map/spray-proposedB/{name}"
Path(pathB).mkdir(parents=True, exist_ok=True)

checkpointA = "checkpoint/spray/ori_proposed-loss=0.4486_dice=0.3202.pth"
checkpointB = "checkpoint/spray/ori_gf_proposed-loss=0.5225_dice=0.4854.pth"

'''Remember to change the weight type of euclidean or proposed at line 54 of model.unet.py'''
modelA = load_model(channel=3, classes=5, graph=True, graph_message_passing=True, model_path=checkpointA).to(device=device)
modelB = load_model(channel=3, classes=5, graph=True, graph_message_passing=True, model_path=checkpointB).to(device=device)

outputsA = get_model_outputs(use_graph=True, model=modelA, image_tensor=image_tensor)
outputsB = get_model_outputs(use_graph=True, model=modelB, image_tensor=image_tensor)

for num_block in range(len(outputsA)):
    xA = outputsA[num_block].squeeze(0)
    yA = torch.sum(xA,0)
    
    xB = outputsB[num_block].squeeze(0)
    yB = torch.sum(xB,0)
    
    plt.imshow(yA.detach().cpu().numpy(), cmap='hot')# gray | hot
    plt.axis('off')
    plt.savefig(f"{pathA}/{num_block + 1}.png", pad_inches = 0, bbox_inches='tight', dpi=300);

    plt.imshow(yB.detach().cpu().numpy(), cmap='hot')# gray | hot
    plt.axis('off')
    plt.savefig(f"{pathB}/{num_block + 1}.png", pad_inches = 0, bbox_inches='tight', dpi=300);


'''
adj_matrix
'''

path = f"output/adjacency_matrix/{name}"
Path(path).mkdir(parents=True, exist_ok=True)

''' euclidean-A '''
feature_mapA = outputsA[4]
batch_size, channels, height, width = feature_mapA.shape
conv1x1 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1).to(device=device)
output_feature_map = conv1x1(feature_mapA)
output_numpy = output_feature_map.detach().cpu().numpy()
output_reshape = np.reshape(output_numpy, (height, width))
adjacency_matrix = proposed(output_reshape, connectivity=8, weight='euclidean')

plt.imshow(adjacency_matrix, cmap=reversed_map, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(f"{path}/euclidean-A.png", pad_inches = 0, bbox_inches='tight', dpi=300);

''' euclidean-B '''
feature_map = outputsB[4]
output_feature_map = conv1x1(feature_map)
output_numpy = output_feature_map.detach().cpu().numpy()
output_reshape = np.reshape(output_numpy, (height, width))
adjacency_matrix = proposed(output_reshape, connectivity=8, weight='euclidean')

plt.imshow(adjacency_matrix, cmap=reversed_map, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(f"{path}/euclidean-B.png", pad_inches = 0, bbox_inches='tight', dpi=300);

''' proposed-A '''
feature_mapA = outputsA[4]
batch_size, channels, height, width = feature_mapA.shape
conv1x1 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1).to(device=device)
output_feature_map = conv1x1(feature_mapA)
output_numpy = output_feature_map.detach().cpu().numpy()
output_reshape = np.reshape(output_numpy, (height, width))
adjacency_matrix = proposed(output_reshape, connectivity=8, weight='proposed')

plt.imshow(adjacency_matrix, cmap=reversed_map, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(f"{path}/proposed-A.png", pad_inches = 0, bbox_inches='tight', dpi=300);

''' proposed-B '''
feature_map = outputsB[4]
output_feature_map = conv1x1(feature_map)
output_numpy = output_feature_map.detach().cpu().numpy()
output_reshape = np.reshape(output_numpy, (height, width))
adjacency_matrix = proposed(output_reshape, connectivity=8, weight='proposed')

plt.imshow(adjacency_matrix, cmap=reversed_map, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(f"{path}/proposed-B.png", pad_inches = 0, bbox_inches='tight', dpi=300);
