from PIL import Image

def crop_image_pillow(image_path, left, upper, right, lower, output_path):
    """
    Crops an image using Pillow based on specified coordinates.

    Args:
        image_path (str): Path to the input image.
        left (int): X-coordinate of the upper-left corner.
        upper (int): Y-coordinate of the upper-left corner.
        right (int): X-coordinate of the lower-right corner (exclusive).
        lower (int): Y-coordinate of the lower-right corner (exclusive).
        output_path (str): Path to save the cropped image.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        print(f"Original image size: {img.size}")

        # Define the cropping box
        crop_box = (left, upper, right, lower)
        print(f"Cropping box: {crop_box}")

        # Perform the crop
        cropped_img = img.crop(crop_box)
        print(f"Cropped image size: {cropped_img.size}")

        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Image successfully cropped and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
