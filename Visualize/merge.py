
import pandas as pd
import numpy as np


# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "/home/labntc/hoaian/Embedding_2D3D/doc/Input.xlsx"
METADATA_FILE = "/home/labntc/hoaian/Embedding_2D3D/doc/Metadata.xlsx"

MERGE_KEY = "usecases"


def normalize_col(c):
    return (
        c.strip()
        .lower()
        .replace("\n", " ")
        .replace("  ", " ")
    )



def merge_file():
    df = pd.read_excel(INPUT_FILE)
    meta = pd.read_excel(METADATA_FILE)

    df.columns = [normalize_col(c) for c in df.columns]
    meta.columns = [normalize_col(c) for c in meta.columns]
    
    meta["usecases"] = (
    meta["usecases"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.rstrip()
    .str.replace("\n", "", regex=True)
    .str.replace("\r", "", regex=True)
    )
    df["usecases"] = (
    df["usecases"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace("\n", "", regex=True)
    .str.replace("\r", "", regex=True)
    .str.rstrip()
    )
    

    df = df.drop(columns=['industry'])
    meta = meta.drop(columns=['industries', 'benefit', 'vertical (sub industry)', 'features'])
    
    df = df.merge(
        meta,
        on="usecases",
        how="left",
    )
    
    
    df["blueprints"] = df["blueprints"].fillna("other")
    df["workload"] = df["workload"].fillna("other")
    df["software - microservices, sdks, libraries, frameworks"] = df["software - microservices, sdks, libraries, frameworks"].fillna("other")
    df["accelerated computing, infra software, networking"] = df["accelerated computing, infra software, networking"].fillna("other")
    df["key isvs and partners"] = df["key isvs and partners"].fillna("other")
    
    
    usecases = str(input("usecases: "))
    description = str(input("description: "))
    benefit = str(input("benefit: "))
    
    new_row = pd.DataFrame([{
    "usecases": usecases,
    "description": description,
    "benefit": benefit
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)
    
    df["usecases"] = (df["usecases"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.rstrip()
    .str.replace("\n", "", regex=True)
    .str.replace("\r", "", regex=True)
    )
    df["benefit"] = (df["benefit"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.rstrip()
    .str.replace("\n", "", regex=True)
    .str.replace("\r", "", regex=True)
    )
    df["description"] = (df["description"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.rstrip()
    .str.replace("\n", "", regex=True)
    .str.replace("\r", "", regex=True)
    )
    
    
    return df


# merge_file()

