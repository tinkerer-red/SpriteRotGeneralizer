import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "output_images")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Rotation Angles (0° to 360° in 15° increments)
ROTATION_ANGLES = list(range(0, 361, 15))

# Padding between images in sprite strip
PADDING = -14




def rgb_to_oklab(color):
    """Convert sRGB (0-255) to OKLab while preserving alpha."""
    def gamma_to_linear(c):
        """Convert sRGB gamma-corrected color to linear RGB."""
        return ((c + 0.055) / 1.055) ** 2.4 if c >= 0.04045 else c / 12.92

    r, g, b, a = color  # ✅ Extract alpha channel
    r, g, b = [gamma_to_linear(c / 255.0) for c in (r, g, b)]  # Normalize and gamma correct

    # Convert to LMS space
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # Cube root transformation
    l, m, s = [c ** (1/3) for c in (l, m, s)]

    # Convert to OKLab
    return (
        l * 0.2104542553 + m * 0.7936177850 + s * -0.0040720468,
        l * 1.9779984951 + m * -2.4285922050 + s * 0.4505937099,
        l * 0.0259040371 + m * 0.7827717662 + s * -0.8086757660,
        a  # ✅ Preserve original alpha
    )


def oklab_to_rgb(color):
    """Convert OKLab back to sRGB while preserving alpha."""
    L, a, b, alpha = color  # ✅ Restore alpha channel

    # Convert OKLab to LMS
    l = L + a * 0.3963377774 + b * 0.2158037573
    m = L + a * -0.1055613458 + b * -0.0638541728
    s = L + a * -0.0894841775 + b * -1.2914855480

    # Cube transformation
    l, m, s = [c ** 3 for c in (l, m, s)]

    # Convert to linear RGB
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    # Convert linear RGB to sRGB gamma space
    def linear_to_gamma(c):
        return 1.055 * (c ** (1/2.4)) - 0.055 if c >= 0.0031308 else 12.92 * c

    r, g, b = [max(0, min(255, round(linear_to_gamma(c) * 255))) for c in (r, g, b)]
    return (r, g, b, alpha)  # ✅ Restore alpha



def extract_palette(image, color_space="sRGB"):
    """Extracts the unique color palette from an image, preserving alpha values.
    
    color_space: "sRGB" (default) or "OKLab". Converts palette accordingly.
    """
    data = np.array(image)  # Convert to NumPy array (H, W, 4)
    colors = np.unique(data.reshape(-1, data.shape[2]), axis=0)  # Get unique RGBA colors
    
    if color_space == "OKLab":
        return np.array([rgb_to_oklab(c) for c in colors])  # Convert to OKLab
    return colors  # Default is sRGB


def resample_to_nearest_palette(image, palette, color_space="sRGB"):
    """Resamples an image to match the nearest colors from the given palette.
    
    color_space: "sRGB" (default) or "OKLab". If "OKLab", convert pixels before lookup.
    """
    data = np.array(image)  # Convert image to NumPy array
    tree = KDTree(palette[:, :4])  # Build KD-Tree for fast color lookup (ignore alpha)

    def find_nearest_color(color):
        if color_space == "OKLab":
            _, index = tree.query(rgb_to_oklab(color))  # Convert pixel to OKLab before lookup
            return oklab_to_rgb(palette[index])  # Convert back to sRGB
        else:
            _, index = tree.query(color[:4])  # Find nearest sRGB color
            return tuple(palette[index])  # Return full (R, G, B, A)

    flat_pixels = data.reshape(-1, 4)  # Flattened (N, 4) for RGBA
    new_pixels = np.array([find_nearest_color(pixel) for pixel in flat_pixels])

    new_data = new_pixels.reshape(data.shape)  # Reshape back to original image shape
    return Image.fromarray(new_data.astype(np.uint8), "RGBA")

def sprite_rot(image, angle, downscale_resample):
    """SpriteRot: High-quality rotation using nearest-neighbor, then downscales using specified method."""
    upscale_factor = 8

    # Step 1: Upscale image
    upscaled = image.resize(
        (image.width * upscale_factor, image.height * upscale_factor),
        resample=Image.NEAREST
    )

    # Step 2: Rotate (expand=True ensures the new bounding box size is correct)
    rotated = upscaled.rotate(angle, resample=Image.NEAREST, expand=True)

    # Step 3: Compute the correct downscale size based on rotation result
    downscale_width = rotated.width // upscale_factor
    downscale_height = rotated.height // upscale_factor

    # Step 4: Downscale while preserving the new bounding size using the specified resampling method
    downscaled = rotated.resize((downscale_width, downscale_height), downscale_resample)

    return downscaled

def sprite_rot_srgb(image, angle, downscale_resample):
    """SpriteRot with color resampling in sRGB space."""
    rotated = sprite_rot(image, angle, downscale_resample)
    palette = extract_palette(image, color_space="sRGB")  # Extract sRGB palette
    return resample_to_nearest_palette(rotated, palette, color_space="sRGB")

def sprite_rot_oklab(image, angle, downscale_resample):
    """SpriteRot with color resampling in OKLab space."""
    rotated = sprite_rot(image, angle, downscale_resample)
    palette = extract_palette(image, color_space="OKLab")  # Extract OKLab palette
    return resample_to_nearest_palette(rotated, palette, color_space="OKLab")



# Define all rotation methods with nearest, bilinear, bicubic, and lanczos
def nearest_rot_srgb(image, angle):
    return sprite_rot_srgb(image, angle, Image.NEAREST)

def nearest_rot_oklab(image, angle):
    return sprite_rot_oklab(image, angle, Image.NEAREST)

def bilinear_rot_srgb(image, angle):
    return sprite_rot_srgb(image, angle, Image.BILINEAR)

def bilinear_rot_oklab(image, angle):
    return sprite_rot_oklab(image, angle, Image.BILINEAR)

def bicubic_rot_srgb(image, angle):
    return sprite_rot_srgb(image, angle, Image.BICUBIC)

def bicubic_rot_oklab(image, angle):
    return sprite_rot_oklab(image, angle, Image.BICUBIC)

def lanczos_rot_srgb(image, angle):
    return sprite_rot_srgb(image, angle, Image.LANCZOS)

def lanczos_rot_oklab(image, angle):
    return sprite_rot_oklab(image, angle, Image.LANCZOS)


# -- regular rotate functions

def rotate_nearest(image, angle):
    """Rotate using nearest-neighbor (pixelated, best for pixel art)."""
    return image.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

def rotate_bilinear(image, angle):
    """Rotate using bilinear interpolation (good balance of sharpness & smoothness)."""
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

def rotate_bicubic(image, angle):
    """Rotate using bicubic interpolation (best for smooth gradients & anti-aliasing)."""
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


# Update rotation methods dictionary to include all new functions
ROTATION_METHODS = {
    "nearest": rotate_nearest,
    "bilinear": rotate_bilinear,
    "bicubic": rotate_bicubic,
    "nearest_rot_srgb": nearest_rot_srgb,
    "nearest_rot_oklab": nearest_rot_oklab,
    "bilinear_rot_srgb": bilinear_rot_srgb,
    "bilinear_rot_oklab": bilinear_rot_oklab,
    "bicubic_rot_srgb": bicubic_rot_srgb,
    "bicubic_rot_oklab": bicubic_rot_oklab,
    "lanczos_rot_srgb": lanczos_rot_srgb,
    "lanczos_rot_oklab": lanczos_rot_oklab,
}



def generate_images(image, rotations, method_name, original_name):
    """Generates and returns a list of rotated images based on the provided rotation angles."""
    processed_images = []  # Store processed images

    # **Determine max width & height**
    max_width = 0
    max_height = 0

    for rotated_img in rotations:
        processed_images.append(rotated_img)  # Store for later use
        max_width = max(max_width, rotated_img.width)
        max_height = max(max_height, rotated_img.height)

    return processed_images  # Return list of processed images

def get_max_image_size(image, rotations):
    """Finds the largest width and height among all rotated images."""
    max_width = 0
    max_height = 0

    for method_name, rotation_func in ROTATION_METHODS.items():
        for angle in rotations:
            rotated_img = rotation_func(image, angle)
            max_width = max(max_width, rotated_img.width)
            max_height = max(max_height, rotated_img.height)

    return max_width, max_height

def generate_example_texture_sheet(image, rotations, original_name):
    """Creates a large texture sheet showcasing all rotation methods in a structured grid."""
    max_width, max_height = get_max_image_size(image, rotations)

    num_methods = len(ROTATION_METHODS)
    num_rotations = len(rotations)

    cell_width = max_width
    cell_height = max_height
    text_width = cell_width * 2  # Text takes 2x width
    padding = 10

    # **Compute final sheet dimensions**
    sheet_width = text_width + (cell_width + padding) * num_rotations
    sheet_height = (cell_height + padding) * num_methods

    # **Create a blank image with transparent background**
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    # **Load default font (or replace with a TTF font for custom styling)**
    try:
        font = ImageFont.truetype("arial.ttf", 10)  # ✅ Reduced text size by half
    except IOError:
        font = ImageFont.load_default()

    # **Iterate over all methods**
    for row, (method_name, rotation_func) in enumerate(ROTATION_METHODS.items()):
        y_offset = row * (cell_height + padding)

        # **Draw white rectangle for text background**
        draw.rectangle([(0, y_offset), (text_width, y_offset + cell_height)], fill=(255, 255, 255, 255))

        # **Draw the method name in the first column**
        text_position = (10, y_offset + (cell_height // 2) - 5)
        draw.text(text_position, method_name, fill=(0, 0, 0, 255), font=font)

        # **Generate rotated images**
        for col, angle in enumerate(rotations):
            rotated_img = rotation_func(image, angle)
            x_offset = text_width + col * (cell_width + padding)

            # Compute centering offsets
            x_center = (cell_width - rotated_img.width) // 2
            y_center = (cell_height - rotated_img.height) // 2

            # Paste rotated image (keeps transparency)
            sheet.paste(rotated_img, (x_offset + x_center, y_offset + y_center), rotated_img)

    return sheet

def process_images():
    """Processes images from the input folder and generates a unified texture sheet."""
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ ERROR: Input folder does not exist! {INPUT_FOLDER}")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("⚠️ No images found in the input folder!")
        return
    
    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)
        original_name = os.path.splitext(filename)[0]  # Extract name without extension
        print(f"📷 Processing {filename}")

        image = Image.open(image_path).convert("RGBA")

        # Generate and save the example texture sheet
        texture_sheet = generate_example_texture_sheet(image, ROTATION_ANGLES, original_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"{original_name}_example_texture.png")
        texture_sheet.save(output_path)

    print(f"🎉 All example texture sheets saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    process_images()
