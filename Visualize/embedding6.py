import numpy as np
import pandas as pd
from tritonclient.grpc import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from merge import merge_file

# -----------------------------
# CONFIG
# -----------------------------

TRITON_URL = "localhost:8001"
MODEL_NAME = "nvidia_nv_embedqa_mistral_7b_v2"

def embed():
    df = merge_file()
    
    
    
    
    
    
    # Ghép thành 1 text (rất quan trọng)
    df["text"] = (
        df["usecases"].fillna("").astype(str)
        + " : "
        + df["description"].fillna("").astype(str)
        + " : "
        + df["benefit"].fillna("").astype(str)
    )
    texts = df["text"].tolist()

    client = InferenceServerClient(
        url=TRITON_URL,
        verbose=False
    )

    # print(client.is_server_live())
    # print(client.is_server_ready())
    # print(client.get_model_repository_index())

    cfg = client.get_model_config(MODEL_NAME)
    all_embeddings = []
    all_token_counts = []

    for idx, text in enumerate(texts):
        text_np = np.array([[text]], dtype=object)
        truncate_np = np.array([["END"]], dtype=object)

        inp_text = InferInput("text", text_np.shape, "BYTES")
        inp_text.set_data_from_numpy(text_np)

        inp_truncate = InferInput("truncate", truncate_np.shape, "BYTES")
        inp_truncate.set_data_from_numpy(truncate_np)

        outputs = [
            InferRequestedOutput("embeddings"),
            InferRequestedOutput("token_count")
        ]

        resp = client.infer(
            model_name=MODEL_NAME,
            inputs=[inp_text, inp_truncate],
            outputs=outputs
        )

        embedding = resp.as_numpy("embeddings")[0]
        token_count = resp.as_numpy("token_count")[0]

        all_embeddings.append(embedding)
        all_token_counts.append(token_count)

        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(texts)}")
            
    return all_embeddings, df
    



