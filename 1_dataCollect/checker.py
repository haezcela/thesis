import csv

# Define the CSV file path
input_csv = 'cases_dataset.csv'

# Function to print decisions from the CSV file
def print_decisions(csv_file, num_files=10):
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        decision_index = headers.index('Decision')  # Find index of 'Decision' column
        
        # Print decisions for the first num_files cases
        count = 0
        for row in reader:
            if count >= num_files:
                break
            decision = row[decision_index]
            print(f"Decision for Case {row[0]}: {decision}")
            count += 1

# Print decisions for the first 10 files
print_decisions(input_csv, num_files=10)
