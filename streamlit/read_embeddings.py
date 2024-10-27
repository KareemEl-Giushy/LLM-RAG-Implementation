import pandas as pd
import numpy as np
from torch import tensor
from sentence_transformers import SentenceTransformer


def load_df(embedding_df_save_path: str= "./text_chunks_and_embeddings.csv"):

    embedding_df_load = pd.read_csv(embedding_df_save_path)

    embedding_df_load["embedding"] = embedding_df_load["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32))

    return embedding_df_load

pages_and_chunks = load_df[["sentence_chunk", "page_number", "chunk_word_count"]].to_dict(orient="records")

text_chunk_embeddings_arr = load_df["embedding"].tolist()

text_chunk_embeddings = tensor(np.array(text_chunk_embeddings_arr)).to("gpu")

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2")
