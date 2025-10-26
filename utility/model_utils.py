import torch
import numpy as np
from PIL import Image
from model.unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(channel=3, classes=5, graph=True, graph_message_passing=True, model_path=''):
    # initialise model
    model = UNet(n_channels=channel, n_classes=classes, graph=graph, graph_message_passing=graph_message_passing)
    # load the weight of the trained model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # switch the model to inference model which freeze the weight of the layer
    model.eval();
    return model


def load_image_to_tensor(img_path):
    image = Image.open(img_path).convert('RGB') #f_01340 f_01522
    img_ndarray = np.array(image)
    img_ndarray = img_ndarray[np.newaxis, ...] if img_ndarray.ndim == 2 else img_ndarray.transpose((2, 0, 1))
    img_ndarray = img_ndarray / 255
    image_tensor = torch.from_numpy(img_ndarray).unsqueeze(0).float()
    image_tensor = image_tensor.to(device=device)
    return image_tensor


def get_model_outputs(use_graph, model, image_tensor):
    model_children = list(model.children())
            
    outputs = [model_children[0](image_tensor)]
    
    if use_graph == True:
        for i in range(1, 6):
            outputs.append(model_children[i](outputs[-1]))
        
        outputs.append(model_children[6](outputs[-1], outputs[3]))
        outputs.append(model_children[7](outputs[-1], outputs[2]))
        outputs.append(model_children[8](outputs[-1], outputs[1]))
        outputs.append(model_children[9](outputs[-1], outputs[0]))
        outputs.append(model_children[10](outputs[-1]))
    else:
        for i in range(1, 5):
            outputs.append(model_children[i](outputs[-1]))
        
        outputs.append(model_children[5](outputs[-1], outputs[3]))
        outputs.append(model_children[6](outputs[-1], outputs[2]))
        outputs.append(model_children[7](outputs[-1], outputs[1]))
        outputs.append(model_children[8](outputs[-1], outputs[0]))
        outputs.append(model_children[9](outputs[-1]))
    return outputs
