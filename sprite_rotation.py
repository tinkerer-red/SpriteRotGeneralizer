import os
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
print([attr for attr in dir(Image.Resampling) if not attr.startswith("__")])
# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "output_images")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Rotation Angles (0° to 360° in 15° increments)
ROTATION_ANGLES = list(range(0, 361, 15))

# Padding between images in sprite strip
PADDING = 2

def sprite_rot(image, angle):
    """Imitates the SpriteRot algorithm: Scale → Rotate → Downscale"""
    upscale_factor = 8
    upscaled = image.resize(
        (image.width * upscale_factor, image.height * upscale_factor), Image.NEAREST
    )
    rotated = upscaled.rotate(angle, resample=Image.NEAREST, expand=True)
    downscaled = rotated.resize(image.size, Image.NEAREST)
    return downscaled

def rotate_nearest(image, angle):
    """Rotate using nearest-neighbor (pixelated, best for pixel art)."""
    return image.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

def rotate_bilinear(image, angle):
    """Rotate using bilinear interpolation (good balance of sharpness & smoothness)."""
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

def rotate_bicubic(image, angle):
    """Rotate using bicubic interpolation (best for smooth gradients & anti-aliasing)."""
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

# Dictionary of rotation methods
ROTATION_METHODS = {
    "spriterot": sprite_rot,
    "nearest": rotate_nearest,
    "bilinear": rotate_bilinear,
    "bicubic": rotate_bicubic,
}

def generate_sprite_strip(image, rotations, method_name):
    """Creates a sprite strip ensuring all images are aligned properly with a uniform max size."""
    # **First pass: Determine max width & height**
    max_width = 0
    max_height = 0
    processed_images = []  # Store processed images for second loop

    for rotated_img in rotations:
        processed_images.append(rotated_img)  # Store for later use
        max_width = max(max_width, rotated_img.width)
        max_height = max(max_height, rotated_img.height)

    # **Second pass: Create sprite strip with proper spacing**
    total_width = (max_width + PADDING) * len(processed_images) - PADDING
    sprite_strip = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))

    for i, rotated_img in enumerate(processed_images):
        x_offset = i * (max_width + PADDING)

        # Compute centering offsets
        x_center = (max_width - rotated_img.width) // 2
        y_center = (max_height - rotated_img.height) // 2

        # Paste centered in its allocated cell
        sprite_strip.paste(rotated_img, (x_offset + x_center, y_center), rotated_img)

    # Save the sprite strip
    output_path = os.path.join(OUTPUT_FOLDER, f"sprite_strip_{method_name}.png")
    sprite_strip.save(output_path)
    print(f"✅ Saved sprite strip: {output_path}")

def process_images():
    """Processes images from the input folder and generates sprite strips"""
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ ERROR: Input folder does not exist! {INPUT_FOLDER}")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("⚠️ No images found in the input folder!")
        return
    
    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)
        print(f"📷 Processing {filename}")

        image = Image.open(image_path).convert("RGBA")

        for method_name, rotation_func in ROTATION_METHODS.items():
            rotated_images = [rotation_func(image, angle) for angle in ROTATION_ANGLES]
            generate_sprite_strip(image, rotated_images, method_name)

    print(f"🎉 All sprite strips saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_images()
