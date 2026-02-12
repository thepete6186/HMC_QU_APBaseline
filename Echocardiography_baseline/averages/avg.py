import numpy as np
import os
import re

# --- CONFIGURATION ---
# Replace this with the exact filename you want to parse
FILENAME = ["CNN_2CH.txt"]  # List of files to parse
# ---------------------

def parse_and_average(filename):
    # Dictionary to hold lists of scores for each metric

    print("{filename}")
    data = {
        "Sensitivity": [],
        "Specificity": [],
        "Precision": [],
        "F1-Score": [],
        "F2-Score": [],
        "Accuracy": []
    }

    file_path = os.path.join(os.getcwd(), "output", filename)
    
    # Check if file exists in 'output' folder, otherwise try current directory
    if not os.path.exists(file_path):
        file_path = os.path.join(os.getcwd(), filename)
        
    if not os.path.exists(file_path):
        print(f"Error: Could not find file '{filename}'. Check the path.")
        return

    print(f"Parsing file: {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            for key in data.keys():
                if line.endswith("nan"):
                        print(f"Warning: Found 'nan' for {key} in file '{filename}'. Skipping this entry.")
                        continue
                if line.startswith(f"{key}:"):
 
                    try:
 
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        data[key].append(value)
                    except ValueError:
                        print(f"Skipping malformed line: {line}")

    print("\n--- RESULTS ---")
    print(f"{'Metric':<15} | {'Average':<10} | {'Count':<5}")
    print("-" * 35)

    for key, values in data.items():
        if len(values) > 0:
            avg = np.mean(values)
            print(f"{key:<15} | {avg:<10.2f} | {len(values):<5}")
        else:
            print(f"{key:<15} | {'N/A':<10} | 0")

for i in range(len(FILENAME)):
    parse_and_average(FILENAME[i])
