import os
import json
import re
from pathlib import Path

def limpiar_texto_chat(texto: str) -> str:
    """Elimina los códigos de tiempo de CHAT y limpia espacios extra."""
    if not isinstance(texto, str):
        return ""
    
    # Busca el carácter \x15 o el símbolo , seguido de números y guiones bajos, y el cierre
    texto_limpio = re.sub(r'[\x15][0-9_]+[\x15]', ' ', texto)
    
    # Eliminar espacios dobles que quedan al quitar los códigos
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio

def main():
    # ==========================================
    # CARPETA OBJETIVO (Se van a sobreescribir los archivos aquí)
    # ==========================================
    TARGET_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl" 
    
    # Las columnas exactas a limpiar
    target_keys = ["Text_interviewer_participant", "Text_participant", "Text_interviewer"]

    print(f"Buscando y sobreescribiendo archivos .jsonl en: {TARGET_DIR}")
    print("¡ATENCIÓN: Los archivos originales serán modificados permanentemente!")
    
    target_path = Path(TARGET_DIR)
    archivos_procesados = 0

    # rglob busca recursivamente en todas las subcarpetas
    for filepath in target_path.rglob('*.jsonl'):
        lineas_limpias = []
        
        # 1. LEER y LIMPIAR: Cargamos todo el archivo a memoria
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                
                # Limpiamos solo los campos que nos interesan
                for key in target_keys:
                    if key in data and data[key]:
                        data[key] = limpiar_texto_chat(data[key])
                
                lineas_limpias.append(data)
        
        # 2. SOBREESCRIBIR: Abrimos el mismo archivo y machacamos el contenido
        with open(filepath, 'w', encoding='utf-8') as outfile:
            for data in lineas_limpias:
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
        archivos_procesados += 1
        # Mostramos la ruta relativa para no saturar la pantalla con rutas larguísimas
        print(f"Sobreescrito: {filepath.relative_to(target_path)}")

    print("==================================================")
    print(f"¡Limpieza completada! Se han sobreescrito {archivos_procesados} archivos.")

if __name__ == "__main__":
    main()