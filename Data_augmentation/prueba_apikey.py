from google import genai
from google.genai import types
from pathlib import Path
import PIL.Image
from dotenv import load_dotenv
import os

load_dotenv()

# colocamos nuestra API KEY
api_key = os.getenv("GEMINI_API_KEY") or ""

# crear cliente
cliente = genai.Client(api_key=api_key)

# abrir la imagen
try:
    img = PIL.Image.open(Path("/mnt/beegfs/groups/irgroup/sara_tfg/assets/cookie-theft-picture.ppm"))
    print("Imagen cargada correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo .ppm en la carpeta del proyecto.")

# reglas estrictas del formato CHAT de TalkBank (diálogo INV y PAR)
reglas_chat = """
IMPORTANTE: Genera ÚNICAMENTE un diálogo transcrito entre un investigador (*INV:) y el paciente (*PAR:). NO incluyas cabeceras, cierres, ni formato Markdown.

Reglas obligatorias de formato y códigos CHAT:
1. Usa "*INV:\t" y "*PAR:\t" al inicio de cada línea.
2. Cada línea debe terminar con un signo de puntuación separado por un espacio (ejemplo: " ." o " ?").
3. Aplica estrictamente los siguientes códigos CHAT dentro del texto de *PAR: para simular su habla:
    - Pausas: (.) corta, (..) media, (...) larga.
    - Muletillas/Rellenos: &-um, &-uh, &-eh.
    - Repeticiones: palabra [/] palabra (ej. la [/] la niña).
    - Revisiones: palabra [//] nueva_palabra (ej. el niño [//] el chico).
    - Falsos arranques fonológicos: &+sonido (ej. &+ga galleta).
    - Parafasias semánticas: palabra_errónea [* s] (ej. madre [* s] en vez de niña).
    - Ininteligible: xxx.
    - Circunloquio (al final de la frase, antes del punto): [+ cir] (ej. lo que da luz . [+ cir]).
    - Habla vacía/Anomia (al final de la frase, antes del punto): [+ es] (ej. coge la cosa esa . [+ es]).
"""

# tipos de perfiles a generar + reglas de formato
perfiles = {
    "Sano": f"""Actúa como un paciente cognitivamente sano realizando una prueba neuropsicológica. 
    Describe lo que ves en esta imagen en primera persona.\n\n{reglas_chat}""",
    
    "DCL": f"""Adopta el papel de un paciente con Deterioro Cognitivo Leve (DCL) realizando una prueba médica. 
    Describe la imagen en primera persona.\n\n{reglas_chat}""",
    
    "Demencia": f"""Actúa estrictamente como un paciente con Demencia/Alzheimer realizando un test. 
    Describe la imagen en primera persona.\n\n{reglas_chat}"""
}

for rol, instrucciones in perfiles.items():
    print(f"\n--- Probando perfil: {rol} ---")
    try: 
        response = cliente.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[instrucciones, img] # le pasamos tanto las instrucciones como la imagen
        )
        print(response.text)
    except Exception as e:
        print(f"Error de Gemini: {e}")
