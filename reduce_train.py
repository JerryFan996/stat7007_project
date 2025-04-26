import os
from PIL import Image

# Define input and output directories
input_folder = "chest_xray_lung"
output_folder = "chest_xray_lung_low_res"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the resolution reduction factor (e.g., 2 means reducing resolution by half)
reduction_factor = 2

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
                # Calculate new dimensions
                new_width = img.width // reduction_factor
                new_height = img.height // reduction_factor

                # Resize the image
                img_low_res = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save the low-resolution image to the output folder
                img_low_res.save(output_path)

print(f"Low-resolution images have been saved to '{output_folder}'.")