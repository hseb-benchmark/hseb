import argparse
from datasets import load_dataset, Features, Value, Dataset
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import partial
import faiss
import os
from tqdm import tqdm

selectivities = [10, 90, 100]


def add_index_tags(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]:
    tags = []
    for index in indices:
        doc_tags = []
        for s in selectivities:
            if index % 100 <= s:
                doc_tags.append(s)
        tags.append(doc_tags)
    return {"id": indices, "tag": tags}


def add_index(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]:
    return {"id": indices}


def zip_embed(embeds: np.ndarray, batch: Dict[str, List], indices: List[int]) -> Dict[str, List]:
    return {"embedding": [embeds[i] for i in indices]}


def zip_ground_truth(
    docs: np.ndarray, scores: np.ndarray, selectivity: int, batch: dict[str, list], indices: list[int]
) -> dict[str, list]:
    return {
        f"results_{selectivity}_docs": [docs[i] for i in indices],
        f"results_{selectivity}_scores": [scores[i] for i in indices],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=str, help="path to queries.json")
    parser.add_argument(
        "--queries-sample", required=False, type=int, default=1000, help="number of queries to randomly sample"
    )
    parser.add_argument("--corpus", required=True, type=str, help="path to corpus.json")
    parser.add_argument(
        "--corpus-sample", required=False, type=int, default=100000, help="number of documents to randomly sample"
    )
    parser.add_argument("--model", type=str, required=True, help="SBERT embedding model to use")
    parser.add_argument("--batch-size", type=int, required=False, default=512)
    parser.add_argument("--top-n", type=int, required=False, default=100)
    parser.add_argument("--out", type=str, required=False, default="out")

    args = parser.parse_args()

    schema = Features({"text": Value("string")})
    queries: Dataset = load_dataset("json", data_files={"train": args.queries}, split="train").select_columns("text")
    if args.queries_sample:
        queries = queries.shuffle().take(args.queries_sample)

    queries = queries.map(
        function=add_index,
        with_indices=True,
        batched=True,
        desc="indexing queries",
    )
    corpus: Dataset = load_dataset("json", data_files={"train": args.corpus}, split="train").select_columns("text")
    if args.corpus_sample:
        corpus = corpus.shuffle().take(args.corpus_sample)
    corpus = corpus.map(
        function=add_index_tags,
        with_indices=True,
        batched=True,
        desc="adding selectivity tags",
    )

    model = SentenceTransformer(args.model)
    print("encoding queries")
    query_embeds = model.encode_query(
        sentences=list(queries.to_dict()["text"]),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        device=["cuda:0", "cuda:1"],
    )
    queries = queries.map(
        function=partial(zip_embed, query_embeds),
        with_indices=True,
        batched=True,
        desc="joining query embeddings",
    )
    print("encoding docs")
    corpus_embeds = model.encode_document(
        sentences=list(corpus.to_dict()["text"]),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        device=["cuda:0", "cuda:1"],
    )
    corpus = corpus.map(
        function=partial(zip_embed, corpus_embeds),
        with_indices=True,
        batched=True,
        desc="joining doc embeddings",
    )

    print("Building Flat index")
    faiss.omp_set_num_threads(os.cpu_count())
    index = faiss.IndexFlatIP(corpus_embeds.shape[1])
    index.add(corpus_embeds)
    print("Exact search...")
    size = corpus_embeds.shape[0]
    docs: Dict[int, np.ndarray] = {}
    doc_scores: Dict[int, np.ndarray] = {}
    for selectivity in tqdm(selectivities):
        bitmap = np.zeros(size, dtype=bool)
        targets = [index for index in range(size) if index % 100 <= selectivity]
        bitmap[targets] = True
        packed = np.packbits(bitmap, bitorder="little")
        params = faiss.SearchParameters(sel=faiss.IDSelectorBitmap(packed))
        scores, docs = index.search(query_embeds, args.top_n, params=params)
        queries = queries.map(
            function=partial(zip_ground_truth, docs, scores, selectivity),
            with_indices=True,
            batched=True,
            desc="zipping",
        )

    corpus.to_json(f"{args.out}/{args.model}/{args.corpus_sample}/corpus.jsonl")
    queries.to_json(f"{args.out}/{args.model}/{args.corpus_sample}/queries.jsonl")
