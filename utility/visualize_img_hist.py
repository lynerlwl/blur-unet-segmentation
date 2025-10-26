from PIL import Image
import matplotlib.pyplot as plt

''' no_blur '''
image_path = "data/spray/train/img-ori/img_rgb_0.png"
''' blur '''
image_path = "data/spray/train/img-0.5/img_rgb_b_0.png"

img = Image.open(image_path)
img = img.convert('RGB')
r, g, b = img.split()


# plt.plot(r.histogram(), color='red')
# plt.plot(g.histogram(), color='green')
# plt.plot(b.histogram(), color='blue')

# plt.tight_layout()
# plt.show()
# plt.savefig(f"output/image_histogram/blur-0.png", pad_inches = 0, bbox_inches='tight', dpi=300);


plt.plot(r.histogram()[:-1], color='red')
plt.plot(g.histogram()[:-1], color='green')
plt.plot(b.histogram()[:-1], color='blue')

plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('RGB Histogram of Original Train Image')

plt.tight_layout()
plt.show()
plt.savefig(f"output/image_histogram/no_blur-1.png", pad_inches = 0, bbox_inches='tight', dpi=300);



# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.plot(r.histogram()[:-1], color='red')
# plt.title('Red Channel')

# plt.subplot(1, 3, 2)
# plt.plot(g.histogram()[:-1], color='green')
# plt.title('Green Channel')

# plt.subplot(1, 3, 3)
# plt.plot(b.histogram()[:-1], color='blue')
# plt.title('Blue Channel')

# plt.tight_layout()
# plt.show()
# plt.savefig(f"output/image_histogram/blur-2.png", pad_inches = 0, bbox_inches='tight', dpi=300);

'''
feature map
'''

''' no_blur '''
image_path = "output/conv_feature_map/spray-proposedA/f_01340/hot/1.png"
''' blur '''
image_path = "output/conv_feature_map/spray-proposedB/f_01340/hot/1.png"

img = Image.open(image_path)
img = img.convert('RGB')
r, g, b = img.split()

plt.plot(r.histogram()[:-1], color='red')
plt.plot(g.histogram()[:-1], color='green')
plt.plot(b.histogram()[:-1], color='blue')

plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('RGB Histogram of Feature Map Trained with Blur Images')

plt.tight_layout()
plt.show()
plt.savefig(f"output/image_histogram/blur-fm.png", pad_inches = 0, bbox_inches='tight', dpi=300);


''' no_blur '''
image_path = "output/conv_feature_map/spray-proposedA/f_01340/hot/1.png"
''' blur '''
image_path = "output/conv_feature_map/spray-proposedB/f_01340/hot/1.png"

img = Image.open(image_path).convert('L')

plt.plot(img.histogram()[:-1])

plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Feature Map Trained with Original Images')

plt.tight_layout()
plt.show()
plt.savefig(f"output/image_histogram/unblur-gray-fm.png", pad_inches = 0, bbox_inches='tight', dpi=300);
