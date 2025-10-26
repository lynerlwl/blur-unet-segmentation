import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
from torch_geometric.data import Data
# import pickle

# with open('conv_layer/improved', 'rb') as f:
#     feature_map = (pickle.load(f)) 

# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def euclidean(image):
#     """
#     Generate an adjacency matrix for a grid graph based on the Euclidean distance of the pixels in an image.
    
#     Args:
#     - image: 2D numpy array representing the image.
    
#     Returns:
#     - adjacency_matrix: 2D numpy array representing the adjacency matrix.
#     """
#     rows, cols = image.shape
    
#     # Generate pixel coordinates
#     coordinates = np.indices((rows, cols)).reshape(2, -1).T
    
#     # Compute pairwise Euclidean distances between pixel coordinates
#     distances = pdist(coordinates, 'euclidean')
    
#     # Convert pairwise distances to a square matrix
#     distance_matrix = squareform(distances)
    
#     # Create adjacency matrix based on distance threshold (Euclidean distance <= sqrt(2) for 8-connectivity)
#     adjacency_matrix = (distance_matrix <= np.sqrt(2)).astype(int)
    
#     return adjacency_matrix


# def proposed(image, connectivity=8, weight='proposed'):
#     """
#     Generate an adjacency matrix for a grid graph based on pixel values in the image.
    
#     Parameters:
#         image (numpy.ndarray): 2D numpy array representing the image.
        
#     Returns:
#         numpy.ndarray: Adjacency matrix of the grid graph.
#     """
#     # Get image dimensions
#     height, width = image.shape
    
#     # Create an empty adjacency matrix
#     adjacency_matrix = np.zeros((height * width, height * width))
    
#     if connectivity == 2:
    
#         # Define directions for adjacent pixels (left, right)
#         directions = [(-1, 0), (1, 0)]
    
#     if connectivity == 4:
    
#         # Define directions for adjacent pixels (up, down, left, right)
#         directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
#     elif connectivity == 8:
        
#         # Define directions for adjacent pixels (including diagonal)
#         directions = [
#             (-1, -1), (-1, 0), (-1, 1),
#             (0, -1),           (0, 1),
#             (1, -1),  (1, 0),  (1, 1)
#         ]
    
#     # Iterate over each pixel in the image
#     for y in range(height):
#         for x in range(width):
#             # Get pixel value
#             pixel_value = image[y, x]
            
#             # Calculate index of current pixel in the adjacency matrix
#             current_index = y * width + x
            
#             if weight == 'proposed':
#                 # Check if pixel value is greater than or equal to 0
#                 if pixel_value >= 0:
#                     # Iterate over adjacent pixels (including diagonal)
#                     for dy, dx in directions:
#                         # Calculate coordinates of adjacent pixel
#                         new_y, new_x = y + dy, x + dx
                        
#                         # Check if adjacent pixel is within image boundaries
#                         if 0 <= new_y < height and 0 <= new_x < width:
#                             # Calculate index of adjacent pixel in the adjacency matrix
#                             adjacent_index = new_y * width + new_x
#                             # Connect current pixel to adjacent pixel
#                             adjacency_matrix[current_index, adjacent_index] = pixel_value
    
#     return adjacency_matrix

def proposed(image, connectivity=8, weight='proposed'):
    """
    Generate an adjacency matrix for a grid graph based on pixel values in the image.
    
    Parameters:
        image (numpy.ndarray): 2D numpy array representing the image.
        
    Returns:
        numpy.ndarray: Adjacency matrix of the grid graph.
    """
    # Get image dimensions
    height, width = image.shape
    
    # Create an empty adjacency matrix
    adjacency_matrix = np.zeros((height * width, height * width))
    
    # if connectivity == 2:
    
    #     # Define directions for adjacent pixels (left, right)
    #     directions = [(-1, 0), (1, 0)]
    
    # if connectivity == 4:
    
    #     # Define directions for adjacent pixels (up, down, left, right)
    #     directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
    if connectivity == 8:
        
        # Define directions for adjacent pixels (including diagonal)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
    for y in range(height):
        for x in range(width):
            # Get pixel value
            pixel_value = image[y, x]
            
            # Calculate index of current pixel in the adjacency matrix
            current_index = y * width + x
        
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                
                if 0 <= new_y < height and 0 <= new_x < width:
                    adjacent_index = new_y * width + new_x
                    
                    if weight == 'proposed':
                        if pixel_value >= 0:
                            adjacency_matrix[current_index, adjacent_index] = pixel_value
                    elif weight == 'euclidean':
                        distance = np.sqrt((y - new_y) ** 2 + (x - new_x) ** 2)
                        adjacency_matrix[current_index, adjacent_index] = distance
                        
    return adjacency_matrix

def feature_map_to_graph(feature_map, conn, weight='proposed'):
    # Assuming feature_map is a 4D tensor (batch_size, channels, height, width)
    batch_size, channels, height, width = feature_map.shape
    
    conv1x1 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1).to(device=device)
    output_feature_map = conv1x1(feature_map)
    output_numpy = output_feature_map.detach().cpu().numpy()
    output_reshape = np.reshape(output_numpy, (height, width))

    # Create node features (flatten each channel for each pixel)
    x = feature_map.view(height * width, channels) # shape: (batch_size * height * width, channels)
    
    # row = height, col = width
    adjacency_matrix = proposed(output_reshape, connectivity=conn, weight=weight)
    edge_index = torch.from_numpy(adjacency_matrix).nonzero().t().contiguous()
    edge_weight = adjacency_matrix[adjacency_matrix > 0]

    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)#
    
    return data.to(device=device)

# weight='euclidean'# euclidean proposed
# conn=8
# feature_map = outputs[4]

# Reverse the flattening of node features
# out2 = out.view(1, 1024, 38, 75) 


