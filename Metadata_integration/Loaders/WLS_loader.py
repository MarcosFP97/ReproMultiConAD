import pandas as pd
import json
import os
from .base_loader import BaseMetadataLoader


class WLSLoader(BaseMetadataLoader):

    def __init__(self, excel_path="/mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad/WLS/WLS-data.xlsx"):
        self.excel_path = excel_path

    def load_metadata(self):
        
        # --- Cargar hoja 1 (age, gender) ---
        df1 = pd.read_excel(self.excel_path, sheet_name=0)
        df1.columns = df1.columns.str.strip()
        df1 = df1.rename(columns=str.lower)
        #print(df1.columns.tolist())
        
        # Expected columns: idtlkbnk, age 2011, sex
        df1 = df1[['idtlkbnk', 'age 2011', 'sex']]

        # Mapeamos los valores numéricos asociados al sexo
        sex_map = {1: "Male", 2: "Female"}
        df1['sex'] = df1['sex'].map(sex_map)

        # --- Cargamos hoja 3 (diagnosis from screening threshold) ---
        df3 = pd.read_excel(self.excel_path, sheet_name=2)
        df3.columns = df3.columns.str.strip()
        df3 = df3.rename(columns=str.lower)
        #print(df3.columns.tolist())


        # Expected columns: education y screeningresult con valores N/Y
        df3 = df3[['idtlkbnk', 'education', 'screeningresult']]

        # Mapeamos diagnóstico
        diagnosis_map = {
            'N': 'HC',        # Healthy control
            'Y': 'Dementia'   # Cognitive impairment confirmed
        }
        df3['screeningresult'] = df3['screeningresult'].map(diagnosis_map)

        # --- Merge metadata ---
        df = pd.merge(df1, df3, on='idtlkbnk', how='left')

        # Construimos diccionario de metadata
        metadata = {}

        for _, row in df.iterrows():
            file_id = str(row['idtlkbnk'])[-5:] # Para poner solo los 5 últimos números, tal y como aparece en File_ID

            metadata[file_id] = {
                "Age": row.get("age 2011", "Unknown"),
                "Gender": row.get("sex", "Unknown"),
                "Education": int(row.get("education", "Unknown")),
                "Diagnosis": row.get("screeningresult", "Unknown"),
                "Continent": "North America",
                "Countries": "United States",
            }

        #print(metadata)
        return metadata

if __name__ == "__main__":
    loader = WLSLoader()  
    metadata = loader.load_metadata()
