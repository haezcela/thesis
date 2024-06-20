import os
import re
import shutil

# Define the directory paths
base_folder = 'appeal_criminal_cases'
affirmed_folder = 'affirmed_criminal_appeal_cases'
non_affirmed_folder = 'nonAffirmed_criminal_appeal_cases'
no_decision_folder = 'no_decision_criminal_appeal_cases'

# Create the directories if they do not exist
os.makedirs(affirmed_folder, exist_ok=True)
os.makedirs(non_affirmed_folder, exist_ok=True)
os.makedirs(no_decision_folder, exist_ok=True)

# Function to determine if a case is affirmed or non-affirmed
def is_affirmed(text):
    affirmed_keywords = ["so ordered", "wherefore"]
    affirmation_patterns = ["affirm", "guilt"]
    
    found_keywords = any(keyword in text.lower() for keyword in affirmed_keywords)
    found_patterns = any(re.search(rf"\b{pattern}\b", text, re.IGNORECASE) for pattern in affirmation_patterns)
    
    if found_keywords and found_patterns:
        return True
    elif not (found_keywords or found_patterns):
        return None
    return False

# Initialize counters for affirmed, non-affirmed, and cases with no decision
affirmed_count = 0
non_affirmed_count = 0
no_decision_count = 0

# Process each year folder
for year in range(1998, 2024):  # Adjust the range as needed
    year_folder = os.path.join(base_folder, str(year), 'Appeals')
    print(f"Processing year folder: {year_folder}")

    if os.path.isdir(year_folder):
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            month_folder = os.path.join(year_folder, month)
            print(f"  Processing month folder: {month_folder}")

            if os.path.isdir(month_folder):
                for case_file in os.listdir(month_folder):
                    case_file_path = os.path.join(month_folder, case_file)
                    print(f"    Processing file: {case_file_path}")  # Print the current file being processed

                    if os.path.isfile(case_file_path):
                        try:
                            with open(case_file_path, 'r', encoding='utf-8') as file:
                                text = file.read()
                                if not text:
                                    print(f"Warning: Empty file {case_file_path}")
                                    continue

                                result = is_affirmed(text)
                                if result is None:
                                    shutil.copy(case_file_path, no_decision_folder)
                                    no_decision_count += 1
                                elif result:
                                    shutil.copy(case_file_path, affirmed_folder)
                                    affirmed_count += 1
                                else:
                                    shutil.copy(case_file_path, non_affirmed_folder)
                                    non_affirmed_count += 1
                        except Exception as e:
                            print(f"Error processing file {case_file_path}: {e}")

# Print results
print(f"Number of affirmed cases: {affirmed_count}")
print(f"Number of non-affirmed cases: {non_affirmed_count}")
print(f"Number of cases with no decision: {no_decision_count}")
