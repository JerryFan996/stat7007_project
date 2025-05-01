import os
from PIL import Image

# Define input and output directories
input_folder = "chest_xray_lung"
output_folder = "chest_xray_lung_low_res"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the target resolution
target_width = 224
target_height = 224

# Walk through all subdirectories and files in the input folder
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            input_path = os.path.join(root, filename)

            # Create corresponding subdirectory structure in the output folder
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            output_path = os.path.join(output_subfolder, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image to 224x224
                img_low_res = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                # Save the resized image to the output folder
                img_low_res.save(output_path)

print(f"Images have been resized to {target_width}x{target_height} and saved to '{output_folder}'.")