import os
import shutil
import re
from tqdm import tqdm

# Define the year to process
year = 2023

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def classify_cases_by_type(source_folder, target_folder):
    appeal_criminal_count = 0
    petition_criminal_count = 0

    # Case-insensitive pattern for "appel"
    appel_pattern = re.compile(r"appel", re.IGNORECASE)

    # Calculate the total number of case files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(source_folder))
    pbar = tqdm(total=total_files, desc="Processing Cases")

    # Iterate through each month folder in the specified year folder
    for month in os.listdir(source_folder):
        month_path = os.path.join(source_folder, month)

        if os.path.isdir(month_path):
            # Create corresponding month subfolder in the target folder for appeal criminal cases
            appeal_criminal_month_folder = os.path.join(target_folder, 'Appeals', month)
            create_folder_if_not_exists(appeal_criminal_month_folder)

            # Iterate through each case file in the month folder
            for case_file in os.listdir(month_path):
                case_file_path = os.path.join(month_path, case_file)

                if os.path.isfile(case_file_path):
                    with open(case_file_path, 'r', encoding='utf-8') as file:
                        case_text = file.read()

                    # Check for the presence of the pattern "appel" to determine if it's an appeal case
                    if appel_pattern.search(case_text):
                        shutil.copy(case_file_path, appeal_criminal_month_folder)
                        appeal_criminal_count += 1
                    else:
                        petition_criminal_count += 1

                    # Update the progress bar
                    pbar.update(1)

    pbar.close()
    return appeal_criminal_count, petition_criminal_count

# Define source and target folders based on the year
source_folder = f'criminal_cases/{year}'
target_folder = f'appeal_criminal_cases/{year}'

# Display the year being processed
print(f"Processing year: {year}")

# Ensure the target folder exists
create_folder_if_not_exists(target_folder)
create_folder_if_not_exists(os.path.join(target_folder, 'Appeals'))

# Classify and count the cases by type
appeal_criminal_count, petition_criminal_count = classify_cases_by_type(source_folder, target_folder)

# Display the counts
print(f"Number of appeal criminal cases: {appeal_criminal_count}")
print(f"Number of petition criminal cases: {petition_criminal_count}")
