import os
import random
import shutil

# Define the folder paths
balanced_affirmed_folder = 'balanced_affirmed_criminal_appeal_cases'
non_affirmed_folder = 'nonAffirmed_criminal_appeal_cases'

train_affirmed_folder = 'train_affirmed_criminal_appeal_cases'
test_affirmed_folder = 'test_affirmed_criminal_appeal_cases'
train_non_affirmed_folder = 'train_nonAffirmed_criminal_appeal_cases'
test_non_affirmed_folder = 'test_nonAffirmed_criminal_appeal_cases'

# Create the train and test folders if they do not exist
os.makedirs(train_affirmed_folder, exist_ok=True)
os.makedirs(test_affirmed_folder, exist_ok=True)
os.makedirs(train_non_affirmed_folder, exist_ok=True)
os.makedirs(test_non_affirmed_folder, exist_ok=True)

# Function to split and copy files
def split_and_copy_files(src_folder, train_folder, test_folder, split_ratio=0.8):
    files = [f for f in os.listdir(src_folder) if f.endswith('.txt')]
    random.shuffle(files)
    
    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    test_files = files[split_point:]
    
    for file_name in train_files:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(train_folder, file_name)
        shutil.copy(src_path, dst_path)
    
    for file_name in test_files:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(test_folder, file_name)
        shutil.copy(src_path, dst_path)

# Split and copy affirmed cases
split_and_copy_files(balanced_affirmed_folder, train_affirmed_folder, test_affirmed_folder)

# Split and copy non-affirmed cases
split_and_copy_files(non_affirmed_folder, train_non_affirmed_folder, test_non_affirmed_folder)

print("Files have been successfully divided into training and testing sets.")
