import csv
import logging
import os
import random
import sys
from typing import Dict, List

import datasets
import numpy as np
import torch
from datasets import IterableDataset

from vec2text.run_args import DataArguments
from vec2text.utils import dataset_map_multi_worker, get_num_proc
import csv
import sys
from datasets import IterableDataset
import numpy as np
import torch

input_data = {
    "text": "/teamspace/studios/this_studio/wikipedia_de_filtered_fullarticles.csv",
    "embeddings": [
        "/teamspace/studios/this_studio/retriv_wiki_de/collections/wiki_de-index_sentence_transf-BAAI/bge-m3_title_only_fullarticles/embeddings/chunk_0.npy",
        "/teamspace/studios/this_studio/retriv_wiki_de/collections/wiki_de-index_sentence_transf-BAAI/bge-m3_title_only_fullarticles/embeddings/chunk_1.npy",
        "/teamspace/studios/this_studio/retriv_wiki_de/collections/wiki_de-index_sentence_transf-BAAI/bge-m3_title_only_fullarticles/embeddings/chunk_2.npy"
    ]
}

def load_text(file_path):
    csv.field_size_limit(sys.maxsize)
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            yield {
                "text": row[2]
            }  # 2 title, 3 text  # Assuming the text is in the fourth column

def load_embeddings(paths):
    for path in paths:
        data = np.load(path)
        for row in data:
            yield torch.tensor(row)  # Directly yield as tensor

class CustomIterableDataset(IterableDataset):
    def __init__(self, text_file, embedding_files, batch_size=4):
        self.text_file = text_file
        self.embedding_files = embedding_files
        self.batch_size = batch_size

    def __iter__(self):
        text_iter = load_text(self.text_file)
        embeddings_iter = load_embeddings(self.embedding_files)
        return self.batch_iter(zip(text_iter, embeddings_iter), self.batch_size)

    def batch_iter(self, iterator, batch_size):
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == batch_size:
                yield self.collate(batch)
                batch = []
        if batch:
            yield self.collate(batch)  # yield the last batch if it's not full size

    def collate(self, batch):
        collated_batch = {}
        collated_batch["text"] = [item[0]["text"] for item in batch]
        collated_batch["embeddings"] = torch.stack([item[1] for item in batch])
        return collated_batch

def load_retriv_wiki_de():
    return CustomIterableDataset(
        input_data["text"], input_data["embeddings"], batch_size=4
    )

def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus() -> datasets.Dataset:
    return datasets.load_dataset("jxm/nq_corpus_dpr")


def load_msmarco_corpus() -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex

def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def load_one_million_paired_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-paired-instructions")
    dataset_dict = dataset_map_multi_worker(
        dataset_dict,
        map_fn=create_ompi_ex,
        num_proc=get_num_proc(),
    )

    return dataset_dict["train"]


def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)

    return dataset_dict["train"]


def load_anthropic_toxic_prompts() -> datasets.Dataset:
    d = datasets.load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d


def load_luar_reddit() -> datasets.Dataset:
    d = datasets.load_dataset("friendshipkim/reddit_eval_embeddings_luar")
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    if data_args.dataset_name == "nq":
        raw_datasets = load_nq_dpr_corpus()
        raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name == "msmarco":
        raw_datasets = load_msmarco_corpus()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_paired_instructions":
        raw_datasets = load_one_million_paired_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "retriv_wiki_de":
        raw_datasets = load_retriv_wiki_de()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "luar_reddit":
        all_luar_datasets = load_luar_reddit()
        raw_datasets = datasets.DatasetDict(
            {
                "train": all_luar_datasets["candidates"],
                "validation": all_luar_datasets["queries"],
            }
        )
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test() -> datasets.Dataset:
    return datasets.load_dataset("ag_news")["test"]


def load_xsum_val(col: str) -> datasets.Dataset:
    d = datasets.load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val() -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio")["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val() -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d


def load_python_code_instructions_18k_alpaca() -> datasets.Dataset:
    d = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d


def load_beir_corpus(name: str) -> List[str]:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    #### Download scifact.zip dataset and unzip the dataset
    beir_datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        name
    )
    out_dir = os.path.join(beir_datasets_cache_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Limit each corpus to first 100k documents.
    MAX_N = 100_000

    if name == "cqadupstack":
        full_corpus = []
        for folder in [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        ]:
            corpus, _queries, _qrels = GenericDataLoader(
                data_folder=os.path.join(data_path, folder)
            ).load(split="test")
            full_corpus.extend([k["text"] for k in corpus.values()])
        random.shuffle(full_corpus)
        return full_corpus[:MAX_N]
    else:
        corpus, _queries, _qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        corpus = [k["text"] for k in corpus.values()]
        return corpus[:MAX_N]


def load_beir_dataset(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    dataset_path = os.path.join(cache_path, "emb_inv_beir", name)
    # print(f"loading BEIR dataset: {name}")
    if os.path.exists(dataset_path):
        logging.info("Loading BEIR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading BEIR dataset %s from JSON (slow) at path %s", dataset_path
        )
        corpus = load_beir_corpus(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        os.makedirs(os.path.join(cache_path, "emb_inv_beir"), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        logging.info("Saved BEIR dataset as HF path %s", dataset_path)
    return dataset


def load_beir_datasets() -> datasets.DatasetDict:
    all_beir_datasets = [
        ####### public datasets #######
        "arguana",
        "climate-fever",
        "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
        ####### private datasets #######
        "signal1m",
        "trec-news",
        "robust04",
        "bioasq",
    ]
    return datasets.DatasetDict({k: load_beir_dataset(k) for k in all_beir_datasets})


def load_standard_val_datasets() -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "ag_news": load_ag_news_test(),
        "anthropic_toxic_prompts": load_anthropic_toxic_prompts(),
        "arxiv": load_arxiv_val(),
        "python_code_alpaca": load_python_code_instructions_18k_alpaca(),
        # "xsum_doc": load_xsum_val("document"),
        # "xsum_summ": load_xsum_val("summary"),
        "wikibio": load_wikibio_val(),
    }
    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    return datasets.DatasetDict(d)
