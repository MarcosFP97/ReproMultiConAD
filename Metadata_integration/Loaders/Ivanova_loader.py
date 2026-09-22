import os
import pandas as pd
from .base_loader import BaseMetadataLoader

class IvanovaLoader(BaseMetadataLoader):

    def __init__(self, excel_path="./Ivanova/Ivanova-meta.xlsx"):
        self.excel_path = excel_path

    def load_metadata(self):
        # --- Load Ivanova data ---
        df = pd.read_excel(self.excel_path)

        # Normalize column names (strip spaces and lowercase)
        df.columns = df.columns.str.strip().str.lower()
        print(f"Detected columns: {df.columns.tolist()}")

        # Keep only the columns of interest
        cols_to_keep = ['identifier', 'mmse', 'schooling years']
        df = df[cols_to_keep]

        # --- Build metadata dictionary ---
        metadata = {}

        for _, row in df.iterrows():
            # Use Identifier as key (e.g., MCI-M-76-1)
            file_id = str(row["identifier"]).strip()

            metadata[file_id] = {
                "MMSE": row.get("mmse", "Unknown"),
                "Education": row.get("schooling years", "Unknown"), # Renamed to Education
                "Continent": "Europe",
                "Countries": "Spain",
            }

        print(f"Loaded {len(metadata)} Ivanova records")
        return metadata

if __name__ == "__main__":
    loader = IvanovaLoader()  
    metadata = loader.load_metadata()