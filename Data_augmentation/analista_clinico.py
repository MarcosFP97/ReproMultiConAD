"""
Análisis clínico-lingüístico del corpus Ivanova usando Mistral vía Ollama.

Introduce el dataset JSONL completo en una ventana de contexto de 128k tokens de
Mistral y solicita una caracterización clínica de cómo el deterioro cognitivo se
manifiesta durante la tarea de lectura del Quijote. El resultado se guarda en
informe_clinico_ivanova.txt y se usó para diseñar las reglas zero-shot de prompt_system.py.
"""

import json
import sys
from pathlib import Path
import ollama

def main():
    # ======================================================
    # CONFIGURACIÓN
    # ======================================================
    RUTA_IVANOVA_REAL = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_ivanova.jsonl" 
    MODELO_ANALIZADOR = "mistral-small3.2" 
    
    required_passage = (
        "En un lugar de La Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía "
        "un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. "
        "Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los "
        "sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las "
        "tres partes de su hacienda."
    )

    print(f"--- Iniciando Análisis Clínico-Lingüístico (Dataset Ivanova) ---")

    # 1. LEER EL DATASET
    try:
        with open(RUTA_IVANOVA_REAL, 'r', encoding='utf-8') as f:
            lineas = f.readlines()
            dataset_text = "".join(lineas)
    except Exception as e:
        sys.exit(f"Error al leer el archivo: {e}")

    # 2. PROMPT DE ANÁLISIS CLÍNICO TOTAL
    prompt_analisis = f"""
    Actúa como un experto mundial en Lingüística Clínica y Neurología Cognitiva, especializado en la detección de Alzheimer mediante el análisis del habla.
    
    ESTÍMULO: Los pacientes están leyendo el inicio del Quijote: "{required_passage}"
    FORMATO: Los datos están en JSONL con Diagnóstico (HC, MCI, Dementia), MMSE y el texto con marcas CHAT de TalkBank.
    
    OBJETIVO: 
    Realiza un análisis clínico descriptivo exhaustivo de cómo el deterioro cognitivo afecta a la lectura y producción verbal en este dataset. No te limites sólo a contar marcas CHAT; analiza la esencia del comportamiento humano frente al texto.
    
    INSTRUCCIONES PARA TU RESPUESTA:
    
    1. CARACTERIZACIÓN POR GRUPOS: 
       - Perfil del Control Sano (HC): ¿Cómo es su fluidez? ¿Qué errores "normales" comete?
       - Perfil de Deterioro Leve (MCI): ¿Qué empieza a fallar? Analiza si hay parafasias, vacilaciones o pérdida de ritmo.
       - Perfil de Demencia: Describe el colapso de la lectura. ¿Hay anomia? ¿Sustituciones semánticas? ¿Pérdida de la estructura gramatical?
    
    2. ANÁLISIS DE LA CARGA COGNITIVA: 
       Analiza en qué partes del pasaje (ej. palabras arcaicas como 'astillero', 'adarga' o enumeraciones complejas) se concentran los fallos según el MMSE.
    
    3. REGLAS TÉCNICAS DE GENERACIÓN (Zero-Shot):
       Redacta una lista de instrucciones (formato lista de Python) para que otro LLM genere textos sintéticos que repliquen estos hallazgos clínicos. 
       - Deben incluir reglas sobre longitud de frase, tipo de errores léxicos y uso de marcas CHAT ([/], [//], &-eh) según el rango de MMSE.
       - Prohibido terminantemente el uso de asteriscos (*) o descripciones narrativas. Solo texto de habla real.

    DATASET REAL PARA ANALIZAR:
    {dataset_text}
    """

    # 3. LLAMADA A MISTRAL (128k context)
    try:
        print("Mistral está procesando el corpus completo... Esto puede tardar un poco.")
        response = ollama.chat(
            model=MODELO_ANALIZADOR,
            messages=[{'role': 'user', 'content': prompt_analisis}],
            options={
                "num_ctx": 128000, 
                "temperature": 0.1 
            }
        )
        
        resultado = response['message']['content']
        
        # 4. GUARDAR RESULTADO
        output_file = "informe_clinico_ivanova.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(resultado)
            
        print("\n" + "="*50)
        print(f"¡ANÁLISIS COMPLETADO! Informe guardado en: {output_file}")
        print("="*50)
        print("\nPrimeros 500 caracteres del informe:\n")
        print(resultado[:500] + "...")

    except Exception as e:
        print(f"Error en la comunicación: {e}")

if __name__ == "__main__":
    main()