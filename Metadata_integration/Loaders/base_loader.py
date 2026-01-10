from abc import ABC, abstractmethod


class BaseMetadataLoader(ABC):
    """
    Clase base para los metadata loaders.
    Cada loader debe implementar load_metadata()
    y devolver un diccionario indexado por file_id.
    """

    @abstractmethod
    def load_metadata(self):
        """
        Carga y devuelve metadata en un diccionario.

        Returns
        -------
        dict
            {
                file_id (str): {
                    "Age": int or str,
                    "Gender": str,
                    "Diagnosis": str,
                    ...
                }
            }
        """
        pass
