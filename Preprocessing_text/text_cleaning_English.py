import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner


# Pitt, Lu, Baycrest, VAS, Kempler, WLS, Delware, taukdial_English_train, taukdial_English_test
input_files = [
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/Pitt.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/Lu.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/Baycrest.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/VAS.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/Kempler.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/WLS.jsonl", #TODO
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/Delware.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukadial_English_train.jsonl", #TODO
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukadial_English_test.jsonl" #TODO
]

output_directory = '/mnt/beegfs/groups/irgroup/sara_tfg/jsonl'
output_filename = 'combined_jsonl_English.jsonl'
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()
English_df= pd.read_json(output_directory + output_filename, lines=True)



# Remove Chinese transcript from Taukdial
def remove_zh_language_rows(df):
    return df[df['Languages'] != 'zh']

def clean_diagnosis(df):
    # Remove specific diagnoses
    diagnoses_to_remove = ['Vascular', 'Memory', 'Aphasia', "Pick's", 'Other']
    df = df[~df['Diagnosis'].isin(diagnoses_to_remove)]
    # Remove rows with empty Diagnosis
    df = df[df['Diagnosis'].notna() & (df['Diagnosis'] != '')]
    
    # Rename diagnoses
    df['Diagnosis'] = df['Diagnosis'].replace({
        'Control': 'HC',
        'NC': 'HC',
        'H': 'HC',
        'AD': 'Dementia',
        'PossibleAD': 'Dementia',
        'ProbableAD': 'Dementia',
        'potential dementia': 'Dementia',
        'D': 'Dementia',
        "Alzheimer's": 'Dementia'
    })
    
    return df

def preprocess_text(text):

    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('PAR', '')
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\\x[0-9A-Za-z_]+\\x', '', text) 
    text = re.sub(r'\b\w+:\s*', '', text) 
    text = text.replace('\n', ' ')
    text = text.replace('→', '')
    text = text.replace('(', '').replace(')', '')
    text = re.sub(r'[\\+^"/„]', '', text)
    text = re.sub(r"[_']", '', text)
    text = text.replace('\t', ' ')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('&=laughs', '')
    text = text.replace('&=nods', '')
    text = text.replace('&=coughs', '')
    text = text.replace('&=snaps:tongue', '')
    text = text.replace('<', '').replace('>', '')
    text = text.replace('*', '').replace('&', '')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.,!?;:])\s+\1', r'\1', text)
    text = re.sub(r'(\.\s*){2,}', '.', text)
    if '.' in text:
        text = text.rsplit('.', 1)[0] + '.' 

    return text

English_df = remove_zh_language_rows(English_df)
English_df = clean_diagnosis(English_df)
English_df["Text_interviewer_participant"] = English_df["Text_interviewer_participant"].apply(preprocess_text)

English_df['Text_length'] = English_df['Text_interviewer_participant'].apply(len)

def remove_short_transcripts(df, min_length=60):
    return df[df['Text_length'] > min_length]


English_df = remove_short_transcripts(English_df)

#train_en, test_en = train_test_split(English_df, test_size=0.2,stratify=English_df['Diagnosis'], random_state=42)

#NOTE SPLIT POR PACIENTES - Para evitar que 1 mismo paciente que ha hecho 3 pruebas distintas acabe en train y test a la vez

# EXTRAER PID REAL DEL File_ID
English_df["PID"] = English_df["File_ID"].str.split("-").str[0]

# 1. Obtener lista única de pacientes
unique_pids = English_df["PID"].unique()

# 2. Obtener diagnóstico asociado a cada PID
pid_diagnosis = English_df.groupby("PID")["Diagnosis"].first()

# 3. Split por pacientes, no por transcripts
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42, stratify=pid_diagnosis.loc[unique_pids])  # estratifica por diagnóstico

# 4. Crear los dataframes finales
train_en = English_df[English_df["PID"].isin(train_pids)]
test_en  = English_df[English_df["PID"].isin(test_pids)]

print("Pacientes en train:", len(train_pids))
print("Pacientes en test:", len(test_pids))
print("Transcripts en train:", len(train_en))

print("Transcripts en test:", len(test_en))

# Save train and test datasets as JSONL
train_en.to_json(output_directory + "train_english.jsonl", orient="records", lines=True, force_ascii=False)
test_en.to_json(output_directory + "test_english.jsonl", orient="records", lines=True, force_ascii=False)

