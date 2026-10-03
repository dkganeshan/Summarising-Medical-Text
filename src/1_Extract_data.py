# Import necessary libraries

import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd #pandas to be installed in the environment
from pathlib import Path


# Function to extract radiology reports from XML files and save to CSV
def extract_radiology_reports(folder_path, output_csv):
    # Initialize an empty list to hold the extracted data
    rows = []
    # create a search pattern for XML files in the specified folder
    search_pattern = os.path.join(folder_path, "*.xml")
    # This finds every XML file that matches the search pattern and loops through them one at a time.
    for f in glob.glob(search_pattern):
        # Use a try-except block to handle potential parsing errors
        try:
            tree = ET.parse(f) # Parse the XML file into an ElementTree object
            findings, impression = None, None # Initialize variables to hold the findings and impression text
            
            # Efficiently iterate through nodes
            for abstract in tree.iter("AbstractText"): # Iterate through all "AbstractText" elements in the XML tree
                label = abstract.get("Label")
                if label == "FINDINGS":
                    findings = abstract.text
                elif label == "IMPRESSION":
                    impression = abstract.text
            
            # Append the extracted data if both findings and impression are available
            if findings and impression:
                rows.append({"findings": findings, "impression": impression})
                
        except ET.ParseError: # Handle XML parsing errors gracefully
            print(f"Skipping malformed file: {f}")
            continue

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

# Execute the combined approach
PROJECT_DIR = Path(__file__).resolve().parent.parent
df_reports = extract_radiology_reports(PROJECT_DIR / "data" / "raw_data", PROJECT_DIR / "data" / "processed_data" / "extracted_data.csv")
print(f"Successfully extracted {len(df_reports)} complete reports.")

print(df_reports.shape)

