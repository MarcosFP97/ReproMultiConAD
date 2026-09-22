import os
import pandas as pd
from .base_loader import BaseMetadataLoader


class TAUKADIALTestLoader(BaseMetadataLoader):

    def __init__(self, csv_path="./TAUKADIAL-24-test/test_ground_truth.csv"):
        self.csv_path = csv_path

    def load_metadata(self):
        # --- Load ground truth ---
        df = pd.read_csv(self.csv_path, sep=';')
        df.columns = df.columns.str.strip().str.lower()

        # Expected columns:
        # tkdname, age, sex, mmse, dx
        df = df[['tkdname','mmse','dx']]

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
                "MMSE": row.get("mmse", "Unknown"),
                "Diagnosis": row.get("dx", "Unknown"),
            }

        print(metadata)
        return metadata

if __name__ == "__main__":
    loader = TAUKADIALTestLoader()  
    metadata = loader.load_metadata()
