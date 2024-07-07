import os
import random
import shutil

# Define the folder paths
affirmed_folder = 'affirmed_criminal_appeal_cases'
balanced_affirmed_folder = 'balanced_affirmed_criminal_appeal_cases'

# Create the balanced_affirmed_folder if it does not exist
os.makedirs(balanced_affirmed_folder, exist_ok=True)

# Get all file names in the affirmed folder
affirmed_files = [f for f in os.listdir(affirmed_folder) if f.endswith('.txt')]

# Randomly select 1867 files from the affirmed folder
selected_files = random.sample(affirmed_files, 1867)

# Copy the selected files to the balanced_affirmed_folder
for file_name in selected_files:
    src_path = os.path.join(affirmed_folder, file_name)
    dst_path = os.path.join(balanced_affirmed_folder, file_name)
    shutil.copy(src_path, dst_path)

print(f"Randomly selected 1867 affirmed cases have been copied to '{balanced_affirmed_folder}'.")
