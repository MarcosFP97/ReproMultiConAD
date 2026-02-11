import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from Extracting_data.collection import JSONLCombiner

TFIDF = False

# Pitt, Lu, Baycrest, VAS, Kempler, WLS, Delware, taukdial_English_train, taukdial_English_test
input_files = [
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Pitt.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Lu.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Baycrest.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/VAS.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Kempler.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/WLS.jsonl", 
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Delaware.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_train.jsonl", 
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_test.jsonl" 
]

output_directory = '/mnt/beegfs/groups/irgroup/sara_tfg/jsonl'
output_filename = 'combined_jsonl_English.jsonl'
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()
English_df = pd.read_json(os.path.join(output_directory, output_filename),lines=True)



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
        'Conrol': 'HC',
        'NC': 'HC',
        'H': 'HC',
        'AD': 'Dementia',
        'DM': 'Dementia',
        'PossibleAD': 'Dementia',
        'ProbableAD': 'Dementia',
        'Probable': 'Dementia',
        'potential dementia': 'Dementia',
        'D': 'Dementia',
        "Alzheimer's": 'Dementia'
    })
    
    return df

def clean_gender(df):
    # Normalizar a string y minúsculas
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()

    # Reemplazos estándar
    df["Gender"] = df["Gender"].replace({
        "m": "M",
        "male": "M",
        "f": "F",
        "female": "F",
        "w": "F",
        "nan": "U",
        "none": "U",
        "": "U"
    })

    # Todo lo que no sea M o F -> U
    df.loc[~df["Gender"].isin(["M", "F"]), "Gender"] = "U"

    return df


def preprocess_text(text):
    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text) 
    # Remove qutation and all punctuation marks, in case of TF-IDF, for e5 you should comment out this part.
    if TFIDF :
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
            text = text.rsplit('.', 1)[0] + '.'  # Keep the text before the last period and add the period

    return text


def remove_short_transcripts(df, min_length=60):
    return df[df['Text_length'] > min_length]

# Cleaning
English_df = remove_zh_language_rows(English_df)
English_df = clean_diagnosis(English_df)
English_df = clean_gender(English_df)
print(English_df["Diagnosis"].value_counts())

# Preprocessing del texto : Tanto el texto completo como solo el del participante  
English_df["Text_interviewer_participant"] = English_df["Text_interviewer_participant"].apply(preprocess_text)
English_df['Text_length'] = English_df['Text_interviewer_participant'].apply(len)

English_df["Text_participant"] = English_df["Text_participant"].apply(preprocess_text)
English_df['Text_length'] = English_df['Text_participant'].apply(len)

# Cleaning 2 : eliminar transcripciones demasiado cortas
English_df = remove_short_transcripts(English_df)

# Split 80/20 
train_en, test_en = train_test_split(English_df, test_size=0.2,stratify=English_df['Diagnosis'], random_state=42)

# Save train and test datasets as JSONL
train_en.to_json(output_directory + "/train_english_e5.jsonl", orient="records", lines=True, force_ascii=False)
test_en.to_json(output_directory + "/test_english_e5.jsonl", orient="records", lines=True, force_ascii=False)

