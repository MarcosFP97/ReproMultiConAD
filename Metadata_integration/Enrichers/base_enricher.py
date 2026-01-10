from abc import ABC, abstractmethod


class BaseMetadataEnricher(ABC):
    """
    Clase base para el enriquecimiento de cada dataset específico

    Cada enricher añade metadata faltante al jsonl base creado para la colección
    """

    @abstractmethod
    def enrich(self, info: dict) -> dict:
        """
        Enriquece el jsonl asociado al dataset

        Parameters
        ----------
        info : dict
            Parsed metadata from a .cha file.

        Returns
        -------
        dict
            The enriched metadata dictionary.
        """
        pass

