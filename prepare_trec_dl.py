import ir_datasets
import pandas as pd
from pathlib import Path


DATASETS = {"2019": "msmarco-document/trec-dl-2019/judged",
            "2020": "msmarco-document/trec-dl-2020/judged"}

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_dataset(year, dataset_id):
    # データセットの読み込み
    dataset = ir_datasets.load(dataset_id)

    # クエリの取得
    queries = []
    for query in dataset.queries_iter():
        queries.append({"query_id": query.query_id,
                        "query": query.text})
    queries_df = pd.DataFrame(queries)

    # qrelsの取得
    qrels = []
    for qrel in dataset.qrels_iter():
        qrels.append({"query_id": qrel.query_id,
                      "doc_id": qrel.doc_id,
                      "relevance": qrel.relevance})
    qrels_df = pd.DataFrame(qrels)

    # 文書IDの抽出と対応する文書の取得
    doc_ids = sorted(qrels_df["doc_id"].unique())
    docstore = dataset.docs_store()
    docs_by_id = docstore.get_many(doc_ids)

    documents = []
    for doc_id in doc_ids:
        doc = docs_by_id[doc_id]
        documents.append({"doc_id": doc.doc_id,
                          "url": doc.url,
                          "title": doc.title,
                          "body": doc.body})
    documents_df = pd.DataFrame(documents)

    # 検索用データセットに整形
    data_df = qrels_df.merge(queries_df, on="query_id")
    data_df = data_df.merge(documents_df, on="doc_id")
    data_df = data_df[["query_id", "query", "doc_id", "url", "title", "body", "relevance"]]

    output_path = OUTPUT_DIR / f"trec_dl_{year}.csv"
    data_df.to_csv(output_path, index=False, encoding="utf-8")

    # 各クエリのコーパスサイズを計算
    corpus_sizes = data_df.groupby("query_id").size()

    print(f" Queries: {len(queries_df)}")
    print(f" Corpus size per query: ", f"{corpus_sizes.min():,}-{corpus_sizes.max():,}")
    print(f" Saved to: {output_path}")


def main():
    for year, dataset_id in DATASETS.items():
        prepare_dataset(year, dataset_id)


if __name__ == "__main__":
    main()
