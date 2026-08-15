import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel
from pathlib import Path


DATASETS = {"2019": Path("data/trec_dl_2019.csv"),
            "2020": Path("data/trec_dl_2020.csv")}

MAX_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]

# 入力トークン数に応じてバッチサイズを指定
BATCH_SIZES = {256: 64,
               512: 32,
               1024: 16,
               2048: 8,
               4096: 2,
               8192: 1}

# ランキング上位100件のみを出力
TOP_K = 100
QUERY_MAX_LENGTH = 512

BGE_MODEL_NAME = "BAAI/bge-m3"
JINA_MODEL_NAME = "jinaai/jina-embeddings-v3"

OUTPUT_DIR = Path("ranking_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_ranking(data_df, model, model_type, max_length):
    results = []
    batch_size = BATCH_SIZES[max_length]

    for query_id, query_df in data_df.groupby("query_id"):
        query = query_df["query"].iloc[0]
        # 文書のタイトルと本文を結合してリスト化
        documents = (query_df["title"] + " " + query_df["body"]).tolist()

        if model_type == "bge_m3":
            # BGE-M3では，sparse, dense, multi-vectorの3種類の埋め込みを選択可能
            query_embedding = model.encode([query],
                                           batch_size=1,
                                           max_length=QUERY_MAX_LENGTH)["dense_vecs"]

            document_embeddings = model.encode(documents,
                                               batch_size=batch_size,
                                               max_length=max_length)["dense_vecs"]

        elif model_type == "jina_embeddings_v3":
            # Jina Embeddings v3では，task引数を指定することで，検索用のクエリと文書に適したタスク固有のベクトルを生成可能
            query_embedding = model.encode([query],
                                           task="retrieval.query",
                                           batch_size=1,
                                           max_length=QUERY_MAX_LENGTH)

            document_embeddings = model.encode(documents,
                                               task="retrieval.passage",
                                               batch_size=batch_size,
                                               max_length=max_length)

        # クエリと文書の埋め込みベクトル（L2正規化済み）の内積（＝コサイン類似度）を計算してスコアを取得
        scores = (query_embedding @ document_embeddings.T)[0]
        # argsortでスコアを降順にソートし，TOP_K件のインデックスを取得
        top_indices = np.argsort(-scores)[:TOP_K]

        for rank, index in enumerate(top_indices, start=1):
            row = query_df.iloc[index]

            results.append({"query_id": query_id,
                            "query": query,
                            "rank": rank,
                            "doc_id": row["doc_id"],
                            "score": scores[index],
                            "relevance": row["relevance"]})

    output = pd.DataFrame(results)

    return output


def save_ranking(data_df, model, model_type, year, max_length):
    print(f"{model_type} | TREC DL {year} | "
          f"max_length={max_length} | "
          f"batch_size={BATCH_SIZES[max_length]}")

    ranking_df = run_ranking(data_df, model, model_type, max_length)

    output_path = (OUTPUT_DIR / f"{model_type}_trec_dl_{year}_{max_length}.csv")
    ranking_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved to: {output_path}")


def main():
    # 半精度（16bit形式）でモデルをロードすることで，VRAMの使用量を削減し，より大きなバッチサイズで推論可能
    bge_model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=True)

    for year, input_path in DATASETS.items():
        # keep_default_na=Falseで，空欄を欠損値扱いしない
        data_df = pd.read_csv(input_path, keep_default_na=False)

        for max_length in MAX_LENGTHS:
            save_ranking(data_df,
                         bge_model,
                         "bge_m3",
                         year,
                         max_length)

    # モデルを削除してGPUメモリを解放
    del bge_model
    torch.cuda.empty_cache()

    jina_model = AutoModel.from_pretrained(JINA_MODEL_NAME,
                                           trust_remote_code=True,
                                           torch_dtype=torch.float16).to("cuda")

    for year, input_path in DATASETS.items():
        data_df = pd.read_csv(input_path, keep_default_na=False)

        for max_length in MAX_LENGTHS:
            save_ranking(data_df,
                         jina_model,
                         "jina_embeddings_v3",
                         year,
                         max_length)

    del jina_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
