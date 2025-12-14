from .base_enricher import BaseMetadataEnricher


class WLSEnricher(BaseMetadataEnricher):

    def __init__(self, wls_metadata: dict):
        self.metadata = wls_metadata
        self.total = 0
        self.matched = 0
        self.missing = 0

    def enrich(self, info: dict) -> dict:
        self.total += 1

        file_id = info.get("File_ID")

        if file_id not in self.metadata:
            self.missing += 1
            return info

        meta = self.metadata[file_id]
        self.matched += 1

        info["age"] = meta.get("Age", info["age"])
        info["gender"] = meta.get("Gender", info["gender"])
        info["Diagnosis"] = meta.get("Diagnosis", info["Diagnosis"])
        info["Education"] = meta.get("Education", info["Education"])
        info["Continents"] = meta.get("Continent", info["Continents"])
        info["Countries"] = meta.get("Countries", info["Countries"])

        return info

    def summary(self):
        return {
            "total_files": self.total,
            "matched_metadata": self.matched,
            "missing_metadata": self.missing
        }
