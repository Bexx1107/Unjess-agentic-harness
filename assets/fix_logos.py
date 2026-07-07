"""Crop and fix the browser-rendered logo PNGs."""
from PIL import Image, ImageChops
import numpy as np

def autocrop_with_padding(img, bg_color, padding=30):
    """Auto-crop image to content bounds with padding."""
    # Create background image for comparison
    bg = Image.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    # Convert to grayscale and get bounding box
    gray = diff.convert('L')
    bbox = gray.getbbox()
    if bbox:
        # Add padding
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(img.size[0], bbox[2] + padding)
        bottom = min(img.size[1], bbox[3] + padding)
        return img.crop((left, top, right, bottom))
    return img

# === Dark logo (black on white bg -> black on transparent) ===
dark = Image.open('assets/logo_horizontal.png').convert('RGBA')
# Crop to content
cropped = autocrop_with_padding(dark, (255, 255, 255, 255), padding=30)
# Make white background transparent
data = cropped.getdata()
new_data = []
for pixel in data:
    # If pixel is very close to white, make transparent
    if pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
        new_data.append((0, 0, 0, 0))
    else:
        new_data.append(pixel)
cropped.putdata(new_data)
cropped.save('assets/logo_horizontal.png')
print(f"Dark logo: {cropped.size}")

# === White logo (white on white bg -> white on transparent) ===
white = Image.open('assets/logo_horizontal_white.png').convert('RGBA')
cropped_w = autocrop_with_padding(white, (255, 255, 255, 255), padding=30)
# Make the entire image white-on-transparent
# The screenshot has white bg with white content (invisible)
# We need to take the dark logo shape and make it white
dark_ref = Image.open('assets/logo_horizontal.png').convert('RGBA')
# Create white version from dark: where dark has content (alpha > 0), make it white
white_logo = dark_ref.copy()
data = white_logo.getdata()
new_data = []
for pixel in data:
    if pixel[3] > 0:  # has content
        new_data.append((255, 255, 255, pixel[3]))
    else:
        new_data.append((0, 0, 0, 0))
white_logo.putdata(new_data)
white_logo.save('assets/logo_horizontal_white.png')
print(f"White logo: {white_logo.size}")
print("Done!")
