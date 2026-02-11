import json
import os
from collections import Counter

import re

def normalize_file_id(file_id, dataset_name):
    """
    Normaliza File_ID dependiendo del dataset.
    """
    file_id = str(file_id).strip().lower()
    dataset_name = dataset_name.lower()

    if dataset_name == "delaware":
        # id-1, id-2 → id
        file_id = file_id.split("-")[0]

    elif dataset_name == "kempler":
        # d6, d6cookie, d6something → d6
        match = re.match(r"(d\d+)", file_id)
        if match:
            file_id = match.group(1)
            
    elif dataset_name.startswith("taukadial"):
        # taukadial-029-1 → taukadial-029
        parts = file_id.split("-")
        if len(parts) >= 3:
            file_id = "-".join(parts[:2])

    return file_id


def count_raw_diagnosis(jsonl_path):
    """
    Cuenta diagnósticos por paciente único (File_ID normalizado).
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]

    diagnosis_counter = Counter()
    seen_file_ids = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            
            #if dataset_name.lower().startswith("taukadial"):
                #language = data.get("Languages")
                #if language != "en":
                #    continue

            raw_file_id = data.get("File_ID")
            if not raw_file_id:
                continue

            # NORMALIZACIÓN ESPECÍFICA POR DATASET
            file_id = data.get("File_ID", "MISSING")
            file_id = normalize_file_id(raw_file_id, dataset_name)

            # Si se quiere contar pacientes únicos, descomentar esta línea. Si se quiere contar nº de transcripciones, comentarla
            if file_id in seen_file_ids:
                continue

            seen_file_ids.add(file_id)

            diagnosis = data.get("Diagnosis", "MISSING")
            if diagnosis is None or diagnosis == "":
                diagnosis = "EMPTY"

            diagnosis_counter[str(diagnosis)] += 1

    return diagnosis_counter

def count_gender(jsonl_path):
    """
    Cuenta valores del campo Gender.
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    gender_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            gender = data.get("Gender", "MISSING")

            if gender is None or gender == "":
                gender = "EMPTY"

            gender_counter[str(gender)] += 1

    return gender_counter


def count_MMSE(jsonl_path):
    """
    Cuenta valores del campo MMSE.
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    MMSE_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            MMSE = data.get("MMSE", "MISSING")

            if isinstance(MMSE, (int, float)):
                MMSE = "Value"

            MMSE_counter[str(MMSE)] += 1

    return MMSE_counter

def count_MMSE(jsonl_path):
    """
    Cuenta valores del campo MMSE.
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    MMSE_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            MMSE = data.get("MMSE", "MISSING")

            if isinstance(MMSE, (int, float)):
                MMSE = "Value"

            MMSE_counter[str(MMSE)] += 1

    return MMSE_counter

def count_Moca(jsonl_path):
    """
    Cuenta valores del campo Moca.
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    Moca_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            Moca = data.get("Moca", "MISSING")

            if isinstance(Moca, (int, float)):
                Moca = "Value"

            Moca_counter[str(Moca)] += 1

    return Moca_counter

def count_age(jsonl_path):
    """
    Cuenta valores del campo Age.
    """

    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    age_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            age = data.get("Age", "MISSING")
            
            if isinstance(age, (int, float)):
                age = "Value"

            age_counter[str(age)] += 1

    return age_counter

def count_tasks(jsonl_path, unique_patients=True):
    """
    Cuenta tareas del campo 'Task' (lista) en el jsonl.

    unique_patients=True  -> cuenta 1 vez por paciente (File_ID)
    unique_patients=False -> cuenta por transcripción (cada línea)
    """
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]

    task_counter = Counter()
    seen_file_ids = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Tu filtro de taukadial (si aplica)
            if dataset_name.lower().startswith("taukadial"):
                language = data.get("Languages")
                if language != "en":
                    continue

            file_id = data.get("File_ID")
            if not file_id:
                continue
            file_id = str(file_id)

            if unique_patients:
                if file_id in seen_file_ids:
                    continue
                seen_file_ids.add(file_id)

            tasks = data.get("Task", None)

            # Normalizamos a lista
            if tasks is None or tasks == "":
                tasks = ["MISSING"]
            elif isinstance(tasks, str):
                # por si a veces viene como string único
                tasks = [tasks]
            elif not isinstance(tasks, list):
                # cualquier otra cosa rara
                tasks = [str(tasks)]

            # Contamos cada tarea
            for t in tasks:
                if t is None or t == "":
                    t = "EMPTY"
                task_counter[str(t)] += 1

    return task_counter


def count_dataset_field(jsonl_path):
    """
    Cuenta cuántas transcripciones hay por valor del campo 'Dataset'.
    """
    dataset_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            dataset_value = data.get("Dataset", "MISSING")

            if dataset_value is None or dataset_value == "":
                dataset_value = "EMPTY"

            dataset_counter[str(dataset_value)] += 1

    return dataset_counter


if __name__ == "__main__":

    jsonl_paths = [
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Baycrest.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Delaware.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Ivanova.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Kempler.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Lu.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/PerLA.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Pitt.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_test.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_train.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/VAS.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/WLS.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_english_e5.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_english_e5.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_spanish.jsonl",
        "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_spanish.jsonl"
    ]

    for jsonl_path in jsonl_paths:
        if not os.path.isfile(jsonl_path):
            print(f"\n[WARNING] File not found: {jsonl_path}")
            continue

        counts = count_raw_diagnosis(jsonl_path)
        gender_counts = count_gender(jsonl_path)
        MMSE_counts = count_MMSE(jsonl_path)
        Moca_counts = count_Moca(jsonl_path)
        age_counts = count_age(jsonl_path)
        dataset_counts = count_dataset_field(jsonl_path)
        task_counts = count_tasks(jsonl_path, unique_patients=False)  # o False si quieres por transcripción
        dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]

        print(f"\nCounts for dataset: {dataset_name}")
        print("--------------------------------")
        print(f"\n- Diagnosis count: ")
        for diagnosis, count in counts.items():
            print(f"{diagnosis:20s} : {count}")
        print(f"\n- Gender count: ")
        for gender, count in gender_counts.items():
            print(f"{gender:20s} : {count}")
        print(f"\n- MMSE count: ")
        for MMSE, count in MMSE_counts.items():
            print(f"{MMSE:20s} : {count}")
        print(f"\n- Moca count: ")
        for Moca, count in Moca_counts.items():
            print(f"{Moca:20s} : {count}")
        print(f"\n- Age count: ")
        for age, count in age_counts.items():
            print(f"{age:20s} : {count}")
        print(f"\n- Task count (per patient): ")
        for t, count in task_counts.items():
            print(f"{t:20s} : {count}")
        if dataset_name in ["train_english_e5", "test_english_e5",
                            "train_spanish", "test_spanish"]:
            print(f"\n- Dataset field count: ")
            for ds, count in dataset_counts.items():
                print(f"{ds:20s} : {count}")
