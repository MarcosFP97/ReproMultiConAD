import os
import pandas as pd
from .base_loader import BaseMetadataLoader


class VASLoader(BaseMetadataLoader):

    def __init__(self, excel_path="./VAS/VAS-data.xlsx"):
        self.excel_path = excel_path

    def load_metadata(self):
        # --- Load VAS data ---
        df = pd.read_excel(self.excel_path, nrows=102)
        df.columns = df.columns.str.strip().str.lower()
        print(df.columns.tolist())

        # Expected columns:
        # vas id, age, gender, previous exp, date, in-person/virtual, moca, h/mci/d, gai, gds
        df = df[['vas id', 'age', 'gender', 'date', 'moca', 'h/mci/d']]

        # --- Normalize gender ---
        gender_map = {
            "M": "Male",
            "F": "Female",
        }
        df['gender'] = df['gender'].map(gender_map)

        # --- Normalize diagnosis ---
        diagnosis_map = {
            "H": "HC",
            "MCI": "MCI",
            "D": "DM",
        }
        df['h/mci/d'] = df['h/mci/d']   .map(diagnosis_map)

        # --- Build metadata dictionary ---
        metadata = {}

        for _, row in df.iterrows():
            file_id = str(row["vas id"]).strip().zfill(3) # Pad with zeros to ensure 3 digits

            metadata[file_id] = {
                "Age": row.get("age", "Unknown"),
                "Gender": row.get("gender", "Unknown"),
                "MoCA": row.get("moca", "Unknown"),
                "Diagnosis": row.get("h/mci/d", "Unknown"),
                "Date": str(row.get("date", "Unknown")),
                "Continent": "America",
                "Countries": "USA",
            }

        print(metadata)
        return metadata

if __name__ == "__main__":
    loader = VASLoader()  
    metadata = loader.load_metadata()
