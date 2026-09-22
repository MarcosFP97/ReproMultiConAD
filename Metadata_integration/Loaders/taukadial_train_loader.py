import os
import pandas as pd
from .base_loader import BaseMetadataLoader


class TAUKADIALTrainLoader(BaseMetadataLoader):

    def __init__(self, csv_path="./TAUKADIAL-24-train/groundtruth.csv"):
        self.csv_path = csv_path

    def load_metadata(self):
        # --- Load ground truth ---
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lower()

        # Expected columns:
        # tkdname, age, sex, mmse, dx
        df = df[['tkdname','age','sex','mmse','dx']]

        # --- Map sex ---
        sex_map = {
            'M': 'Male',
            'F': 'Female',
            1: 'Male',
            2: 'Female'
        }
        df['sex'] = df['sex'].map(sex_map)

        # --- Map diagnosis ---
        dx_map = {
            'NC': 'HC',
            'MCI': 'MCI',
        }
        df['dx'] = df['dx'].map(dx_map)

        # --- Build metadata dictionary ---
        metadata = {}

        for _, row in df.iterrows():
            file_id = os.path.splitext(str(row['tkdname']).strip())[0].lower() # Para quitar lo de .wav

            metadata[file_id] = {
                "Age": row.get("age", "Unknown"),
                "Gender": row.get("sex", "Unknown"),
                "MMSE": row.get("mmse", "Unknown"),
                "Diagnosis": row.get("dx", "Unknown"),
            }

        print(metadata)
        return metadata
    
if __name__ == "__main__":
    loader = TAUKADIALTrainLoader()  
    metadata = loader.load_metadata()
