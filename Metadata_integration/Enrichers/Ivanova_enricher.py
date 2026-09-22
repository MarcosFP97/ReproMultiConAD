from .base_enricher import BaseMetadataEnricher
import pandas as pd

class IvanovaEnricher(BaseMetadataEnricher):
    """Enricher for the Ivanova dataset: maps and fills Ivanova-specific metadata."""

    def __init__(self, ivanova_metadata: dict):
        self.metadata = ivanova_metadata
        self.total = 0
        self.matched = 0
        self.missing = 0

    def enrich(self, info: dict) -> dict:
        """Enrich a single parsed metadata dict with Ivanova metadata when available."""

        self.total += 1

        # The File_ID must match the 'identifier' that we store in the Loader
        file_id = info.get("File_ID")

        if file_id not in self.metadata:
            self.missing += 1
            return info 

        meta = self.metadata[file_id]
        self.matched += 1

        # --- Mapping of Ivanova-specific fields ---
        # We use .get(loader_key, default_value_if_missing)
        # --- MMSE to int ---
        mmse_val = meta.get("MMSE", "Unknown")
        if mmse_val != "Unknown" and pd.notna(mmse_val):
            info["MMSE"] = int(float(mmse_val))
        
        # --- Education to int ---
        edu_val = meta.get("Education", "Unknown")
        if edu_val != "Unknown" and pd.notna(edu_val):
            info["Education"] = int(float(edu_val))
        
        # Geographic fields
        info["Continents"] = meta.get("Continent", info.get("Continents", "Europe"))
        info["Countries"] = meta.get("Countries", info.get("Countries", "Russia"))

        return info

    def summary(self):
        return {
            "total_files": self.total,
            "matched_metadata": self.matched,
            "missing_metadata": self.missing
        }