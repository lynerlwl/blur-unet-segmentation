from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2 

def filter_contours_by_aabb(contours, ratio_threshold, arc_length_threshold):
    """
    Filters a list of contours based on the aspect ratio and arc length of their
    axis-aligned bounding box.

    Args:
        contours (list): A list of contours.
        ratio_threshold (float): The maximum allowed aspect ratio (w/h).
        arc_length_threshold (float): The maximum allowed arc length (perimeter).

    Returns:
        list: A new list containing the contours that meet the criteria.
    """
    contours_selected = []
    for cnt in contours:
        # Get the axis-aligned bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate aspect ratio
        if h == 0:  # Avoid division by zero
            ratio = 0
        else:
            ratio = float(w) / h
        
        # Check if the contour meets both criteria
        if ratio < ratio_threshold and cv2.arcLength(cnt, True) <= arc_length_threshold:
            contours_selected.append(cnt)
            
    return contours_selected
    
def filter_contours_by_rotated_bb(contours, ratio_threshold, arc_length_threshold):
    """
    Filters a list of contours based on the aspect ratio and arc length of their
    rotated bounding box.

    Args:
        contours (list): A list of contours.
        ratio_threshold (float): The maximum allowed aspect ratio (longer side/shorter side).
        arc_length_threshold (float): The maximum allowed arc length (perimeter).

    Returns:
        list: A new list containing the contours that meet the criteria.
    """
    contours_selected = []
    for cnt in contours:
        # Get the rotated bounding box (minAreaRect)
        rect = cv2.minAreaRect(cnt)
        
        # The rect variable is a tuple: (center(x, y), size(w, h), angle)
        # where w and h are the dimensions of the rotated rectangle.
        w, h = rect[1]
        
        # Ensure the ratio is always > 1 by dividing the longer side by the shorter
        if h > w:
            w, h = h, w
        
        if h == 0: # Avoid division by zero
            ratio = 0
        else:
            ratio = float(w) / h

        # Check if the contour meets both criteria
        if ratio < ratio_threshold and cv2.arcLength(cnt, True) <= arc_length_threshold:
            contours_selected.append(cnt)

    return contours_selected


'''
isolate classes from mask
'''

# path = "predicted/5_ori_proposed"
path = "data/spray/test"
# target = 'mask-f_01340'#'mask-f_01522' #
target = 'f_01318'#'f_01522'#
msk = Image.open(f"{path}/{target}.png")
img_arr = np.array(msk)

# for class 1 and 2

selected_pixel = img_arr.copy()
selected_pixel[selected_pixel >= 3] = 0
cmap = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green'])
save_name1 = f"{path}/{target}-select.png"
plt.imshow(selected_pixel, cmap=cmap)
plt.axis('off')
plt.savefig(save_name1, bbox_inches='tight', pad_inches = 0, dpi=300);

# to drop and att_lig

class_name = ['droplet', 'detached_ligament']
cmap = colors.LinearSegmentedColormap.from_list("", ['white', 'blue'])
for i in range(0,2):
    selected_pixel = img_arr.copy()
    selected_pixel[selected_pixel != i+1] = 0
    
    if len(np.unique(selected_pixel)) == 2:
        save_name = f"{path}/{target}-{class_name[i]}.png"
        plt.imshow(selected_pixel, cmap=cmap)
        plt.axis('off')
        plt.savefig(save_name, bbox_inches='tight', pad_inches = 0, dpi=300);

'''
estimate number of droplets
'''
save_name1=f"{path}/{target}.png"
save_name1=f"{path}/{target}-select.png"
image = cv2.imread(save_name1)
gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #to grayscale
_, binary = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV) 
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # source image, contour retrieval mode, contour approximation method
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

ratio_threshold = 1.5 #if target == 'mask-f_01340' else 1.5
arc_length_threshold = 75

# contours_selected = []
# for cnt in contours:
#   x, y, w, h = cv2.boundingRect(cnt)
#   ratio = float(w)/h
#   if ratio < ratio_threshold and cv2.arcLength(cnt,True) <= arc_length: contours_selected.append(cnt)

contours_selected = filter_contours_by_aabb(contours, ratio_threshold, arc_length_threshold)
contours_selected = filter_contours_by_rotated_bb(contours, ratio_threshold, arc_length_threshold)

save_name2 = f"{path}/{target}-detached_ligament.png"
image = cv2.imread(save_name2)
gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #to grayscale
_, binary = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV) 
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # source image, contour retrieval mode, contour approximation method

# contours_selected2 = []
# for cnt in contours:
#   x, y, w, h = cv2.boundingRect(cnt)
#   ratio = float(w)/h
#   if ratio <= ratio_threshold and 0.01 <= cv2.arcLength(cnt,True) <= arc_length_threshold: contours_selected2.append(cnt)

contours_selected2 = filter_contours_by_aabb(contours, ratio_threshold, arc_length_threshold)
# contours_selected2 = filter_contours_by_rotated_bb(contours, ratio_threshold, arc_length_threshold)

total_droplets = 1088 if target == 'mask-f_01340' else 1495 
detected = len(contours_selected) 
detected_rate = len(contours_selected) / total_droplets * 100
detected_wrong = len(contours_selected2)
detected_wrong_rate = detected_wrong / detected * 100
detected_correct = detected - detected_wrong
detected_correct_rate = detected_correct / total_droplets * 100
print(f"""All droplets detected = {detected} / {total_droplets} = {detected_rate:.2f}%
Droplets detected as detached_ligaments = {detected_wrong} / {detected} = {detected_wrong_rate:.2f}%
Droplets detected correctly = {detected_correct} / {total_droplets} = {detected_correct_rate:.2f}%""")
      
# for index, cnt in enumerate(contours):
#     x = cnt.ravel()[0]
#     y = cnt.ravel()[1] - 5
#     cv2.putText(image, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

# cv2.drawContours(image=image, contours=contours_selected, contourIdx=-1, color=(0,0,0), thickness=1)
# plt.imshow(image)
# plt.axis('off')
# plt.savefig(f"temp/all_cnt_h.png", bbox_inches='tight', pad_inches = 0, dpi=300);

