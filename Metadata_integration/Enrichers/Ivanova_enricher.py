from .base_enricher import BaseMetadataEnricher
import pandas as pd

class IvanovaEnricher(BaseMetadataEnricher):

    def __init__(self, ivanova_metadata: dict):
        self.metadata = ivanova_metadata
        self.total = 0
        self.matched = 0
        self.missing = 0

    def enrich(self, info: dict) -> dict:
        self.total += 1

        # El File_ID debe coincidir con el 'identifier' que guardamos en el Loader
        file_id = info.get("File_ID")

        if file_id not in self.metadata:
            self.missing += 1
            return info 

        meta = self.metadata[file_id]
        self.matched += 1

        # --- Mapeo de campos específicos de Ivanova ---
        # Usamos .get(llave_del_loader, valor_por_defecto_si_no_existe)
        # --- MMSE a int ---
        mmse_val = meta.get("MMSE", "Unknown")
        if mmse_val != "Unknown" and pd.notna(mmse_val):
            info["MMSE"] = int(float(mmse_val))
        
        # --- Education a int ---
        edu_val = meta.get("Education", "Unknown")
        if edu_val != "Unknown" and pd.notna(edu_val):
            info["Education"] = int(float(edu_val))
        
        # Campos geográficos
        info["Continents"] = meta.get("Continent", info.get("Continents", "Europe"))
        info["Countries"] = meta.get("Countries", info.get("Countries", "Russia"))

        return info

    def summary(self):
        return {
            "total_files": self.total,
            "matched_metadata": self.matched,
            "missing_metadata": self.missing
        }