from .base_enricher import BaseMetadataEnricher


class TAUKADIALTestEnricher(BaseMetadataEnricher):

    def __init__(self, taukadial_test_metadata: dict):
        self.metadata = taukadial_test_metadata
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

        info["Diagnosis"] = meta.get("Diagnosis", info["Diagnosis"])
        info["MMSE"] = meta.get("MMSE", info["MMSE"])

        return info

    def summary(self):
        return {
            "total_files": self.total,
            "matched_metadata": self.matched,
            "missing_metadata": self.missing
        }
