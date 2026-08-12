import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path

DATASETS = {"2019": Path("data/trec_dl_2019.csv"),
            "2020": Path("data/trec_dl_2020.csv")}

TOKENIZER_NAME = "BAAI/bge-m3"
OUTPUT_DIR = Path("statistics")
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_statistics(data_df, tokenizer):
    document_lengths = []

    for _, row in data_df.iterrows():
        document = row['title'] + " " + row['body']
        document_lengths.append(len(tokenizer.encode(document)))

    data_df = data_df.copy()
    data_df["document_length"] = document_lengths

    statistics = []

    for query_id, query_df in data_df.groupby("query_id"):
        relevance_counts = (query_df["relevance"].value_counts().reindex([0, 1, 2, 3], fill_value=0))

        lengths = query_df["document_length"]

        statistics.append({"query_id": query_id,
                           "query": query_df["query"].iloc[0],
                           "document_count": len(query_df),
                           "relevance_0": relevance_counts[0],
                           "relevance_1": relevance_counts[1],
                           "relevance_2": relevance_counts[2],
                           "relevance_3": relevance_counts[3],
                           "min_document_length": lengths.min(),
                           "max_document_length": lengths.max(),
                           "mean_document_length": lengths.mean(),
                           "variance_document_length": lengths.var(ddof=0)})

    return pd.DataFrame(statistics)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for year, input_path in DATASETS.items():
        data_df = pd.read_csv(input_path)
        statistics_df = calculate_statistics(data_df, tokenizer)

        output_path = OUTPUT_DIR / f"trec_dl_{year}_statistics.csv"
        statistics_df.to_csv(output_path, index=False, encoding="utf-8")

        print(f" Queries: {len(statistics_df):,}")
        print(f" Saved to: {output_path}")


if __name__ == "__main__":
    main()

