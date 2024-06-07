import os
import shutil
import re
from tqdm import tqdm

# Define the year to process
year = 2000

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def classify_cases(source_folder, target_folder):
    criminal_count = 0
    civil_count = 0

    # Case-insensitive pattern for "People of the Phil"
    pattern = re.compile(r"people of the phil", re.IGNORECASE)

    # Calculate the total number of case files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(source_folder))
    pbar = tqdm(total=total_files, desc="Processing Cases")

    # Iterate through each month folder in the specified year folder
    for month in os.listdir(source_folder):
        month_path = os.path.join(source_folder, month)

        if os.path.isdir(month_path):
            # Create corresponding month subfolder in the target folder for criminal cases
            criminal_month_folder = os.path.join(target_folder, month)
            create_folder_if_not_exists(criminal_month_folder)

            # Iterate through each case file in the month folder
            for case_file in os.listdir(month_path):
                case_file_path = os.path.join(month_path, case_file)

                if os.path.isfile(case_file_path):
                    with open(case_file_path, 'r', encoding='utf-8') as file:
                        case_text = file.read()

                    if pattern.search(case_text):
                        # Copy the criminal case to the new folder
                        shutil.copy(case_file_path, criminal_month_folder)
                        criminal_count += 1
                    else:
                        civil_count += 1

                    # Update the progress bar
                    pbar.update(1)

    pbar.close()
    return criminal_count, civil_count

# Define source and target folders based on the year
source_folder = f'cases/{year}'
target_folder = f'criminal_cases/{year}'

# Display the year being processed
print(f"Processing year: {year}")

# Ensure the target folder exists
create_folder_if_not_exists(target_folder)

# Classify and count the cases
criminal_count, civil_count = classify_cases(source_folder, target_folder)

# Display the counts
print(f"Number of criminal cases: {criminal_count}")
print(f"Number of civil cases: {civil_count}")
