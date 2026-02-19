import os
import pandas as pd
from .base_loader import BaseMetadataLoader

class IvanovaLoader(BaseMetadataLoader):

    def __init__(self, excel_path="/mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad/Ivanova/Ivanova-meta.xlsx"):
        self.excel_path = excel_path

    def load_metadata(self):
        # --- Cargar datos de Ivanova ---
        df = pd.read_excel(self.excel_path)
        
        # Normalizar nombres de columnas (quitar espacios y poner en minúsculas)
        df.columns = df.columns.str.strip().str.lower()
        print(f"Columnas detectadas: {df.columns.tolist()}")

        # Seleccionamos solo las columnas de interés
        cols_to_keep = ['identifier', 'mmse', 'schooling years']
        df = df[cols_to_keep]

        # --- Construir el diccionario de metadatos ---
        metadata = {}

        for _, row in df.iterrows():
            # Usamos el Identifier como clave (ej: MCI-M-76-1)
            file_id = str(row["identifier"]).strip()

            metadata[file_id] = {
                "MMSE": row.get("mmse", "Unknown"),
                "Education": row.get("schooling years", "Unknown"), # Renombrado a Education
                "Continent": "Europe",
                "Countries": "Spain",
            }

        print(f"Cargados {len(metadata)} registros de Ivanova")
        return metadata

if __name__ == "__main__":
    loader = IvanovaLoader()  
    metadata = loader.load_metadata()