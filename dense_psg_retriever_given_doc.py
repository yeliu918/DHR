#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dhr.data.qa_validation import calculate_matches, calculate_doc_matches
from dhr.models import init_biencoder_components
from dhr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dhr.utils.data_utils import Tensorizer
from dhr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

from dhr.indexer.faiss_indexers_old import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)
import sys

csv.field_size_limit(sys.maxsize)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
num_gpu = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpu))
print(torch.cuda.is_available())


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
            self,
            question_encoder: nn.Module,
            batch_size: int,
            tensorizer: Tensorizer,
            index: DenseIndexer,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):
                ## question length is short and we set the maximum length as 80 tokens
                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)[0:80]
                    for q in questions[batch_start: batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        print("Begin to concatenate the query vectors............")
        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info("Total encoded queries tensor %s", query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
            self, query_vectors: np.array, top_docs: int = 100
    ):  # -> List[Tuple[List[object], List[float]]], List
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        # logger.info("index search time: %f sec.", time.time() - time0)
        used_time = time.time() - time0
        return results, used_time


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def parse_qa_json_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = json.load(ifile)
        for row in reader:
            question = row["question"]
            answers = row["short_answers"]
            title = row["title"]
            yield question, answers, title


def parse_qa_json_file_with_psgs_output(location1, location2, topk) -> Iterator[Tuple[str, List[str]]]:
    with open(location1) as ifile:
        with open(location2) as ipsg_gold:
            reader = csv.reader(ifile, delimiter="\t")
            psg_gold = json.load(ipsg_gold)
            for row, psg in zip(reader, psg_gold):
                question1 = row[0]
                question2 = psg["question"]
                if question1 == question2:
                    answers = eval(row[1])
                    retrieval_title = []
                    title_score = {}
                    for ctxs in psg["ctxs"][:topk]:
                        retrieval_title.append(ctxs['title'])
                        title_score[ctxs['title']] = ctxs['score']
                    yield question1, answers, retrieval_title, title_score


def validate(
        passages: Dict[object, Tuple[str, str]],
        answers: List[List[str]],
        result_ctx_ids: List[Tuple[List[object], List[float]]],
        workers_num: int,
        match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    if len(top_k_hits) < 100:
        logger.info("Top-1:  %s  Top-5:  %s  Top-20:  %s   ", top_k_hits[0], top_k_hits[4], top_k_hits[19])
    elif len(top_k_hits) == 100:
        logger.info("Top-1:  %s  Top-5:  %s  Top-20:  %s  Top-100:  %s  ", top_k_hits[0], top_k_hits[4], top_k_hits[19],
                    top_k_hits[99])
    return match_stats.questions_doc_hits


def validate_doc(
        passages: Dict[object, Tuple[str, str]],
        id_title: Dict[object, str],
        title_text: Dict[str, str],
        answers: List[List[str]],
        result_ctx_ids: List[Tuple[List[object], List[float]]],
        workers_num: int,
        match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_doc_matches(
        id_title, title_text, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    if len(top_k_hits) < 100:
        logger.info("Top-1:  %s  Top-5:  %s  Top-20:  %s   ", top_k_hits[0], top_k_hits[4], top_k_hits[19])
    elif len(top_k_hits) == 100:
        logger.info("Top-1:  %s  Top-5:  %s  Top-20:  %s  Top-100:  %s  ", top_k_hits[0], top_k_hits[4], top_k_hits[19],
                    top_k_hits[99])
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str):
    output_docs = {}
    id_title = {}
    title_text = {}
    title_id = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    output_docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    if len(row) == 3:
                        output_docs[row[0]] = (row[1], row[2])
                        id_title[row[0]] = row[2]
                        if row[2] not in title_text.keys():
                            title_text[row[2]] = [row[1]]
                        else:
                            title_text[row[2]].append(row[1])
                    elif len(row) == 4:
                        output_docs[row[0]] = (row[1], row[2], row[3])
                        id_title[row[0]] = row[2]
                        if row[2] not in title_text.keys():
                            title_text[row[2]] = [row[1]]
                            title_id[row[2]] = [row[0]]
                        else:
                            title_text[row[2]].append(row[1])
                            title_id[row[2]].append(row[0])
    return output_docs, id_title, title_id, title_text


def save_results(
        passages: Dict[object, Tuple[str, str]],
        questions: List[str],
        answers: List[List[str]],
        top_passages_and_scores: List[Tuple[List[object], List[float]]],
        per_question_hits: List[List[bool]],
        out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "title_list": docs[c][2],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")

    prefix_len = len("question_model.")
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("question_model.")
    }

    # if "embeddings.position_ids" in question_encoder_state:
    #     question_encoder_state.pop("embeddings.position_ids")
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # get questions & answers
    questions = []
    question_answers = []
    topk_doc_title = []
    all_title_score = []
    if args.retrieval_doc_file:
        for ds_item in parse_qa_json_file_with_psgs_output(args.qa_file, args.retrieval_doc_file, topk=args.topk_doc):
            question, answers, title, title_score = ds_item
            questions.append(question)
            question_answers.append(answers)
            topk_doc_title.append(title)
            all_title_score.append(title_score)
    else:
        for ds_item in parse_qa_json_file(args.qa_file):
            question, answers, title = ds_item
            questions.append(question)
            question_answers.append(answers)
            topk_doc_title.append(title)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (
            os.path.exists(index_path) or os.path.exists(index_path + ".index.dhr")
    ):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info("Reading all passages data from files: %s", input_paths)

        # retriever.index.index_data(input_paths)
        def read_data(vector_files: List[str]):
            buffer = {}
            for i, item in enumerate(iterate_encoded_files(vector_files)):
                db_id, doc_vector = item
                buffer[db_id] = doc_vector
            logger.info("Total data read %d", len(buffer.keys()))
            logger.info("Data finish reading.")
            return buffer

        buffer = read_data(input_paths)

        if args.save_or_load_index:
            retriever.index.serialize(index_path)

    all_passages, id_title, title_id, title_text = load_passages(args.ctx_file)

    print("Generating the question tensor............")
    questions_tensor = retriever.generate_question_vectors(questions)
    print("Finish generating the question tensor............")

    used_time_list = []
    top_ids_and_scores = []
    for question_embd, title, title_score in zip(questions_tensor, topk_doc_title, all_title_score):
        if isinstance(title, List):
            ids = []
            doc_score_dic = {}
            for tit in title:
                new_id = title_id[tit]
                ids.extend(new_id)
                for id in new_id:
                    doc_score_dic[id] = float(title_score[tit])
            assert list(doc_score_dic.keys()) == ids
            buffer_gold = [(id, buffer[id]) for id in ids]
            retriever.index.index_data_with_restricted(buffer_gold)
            top_ids_and_score, used_time = retriever.get_top_docs(question_embd.numpy().reshape(1, -1),
                                                                  len(buffer_gold))
            top_ids_and_score = top_ids_and_score[0]
            top_ids_and_score_dic = {ids: score for ids, score in zip(top_ids_and_score[0], list(top_ids_and_score[1]))}
            new_top_ids_and_score_dic = {kid: top_ids_and_score_dic[kid] + args.lamd * doc_score_dic[kid] for kid in
                                         top_ids_and_score_dic.keys()}
            sorted_ids_list = [k for k, v in sorted(new_top_ids_and_score_dic.items(), key=lambda item: -item[1])]
            sorted_scores = [new_top_ids_and_score_dic[ids] for ids in sorted_ids_list]
            top_ids_and_score = (sorted_ids_list[0:args.topk_psgs], np.array(sorted_scores)[0:args.topk_psgs])
            top_ids_and_scores.append(top_ids_and_score)
            used_time_list.append(used_time)
        elif isinstance(title, str):
            ids = title_id[title]
            buffer_gold = [(id, buffer[id]) for id in ids]
            retriever.index.index_data_with_restricted(buffer_gold)
            top_ids_and_score = retriever.get_top_docs(question_embd.numpy().reshape(1, -1), len(buffer_gold))
            top_ids_and_scores.append(top_ids_and_score[0])

    logger.info("total index search time: %f sec.", sum(used_time_list))
    logger.info("average index search time: %f sec.", sum(used_time_list) / len(used_time_list))

    if len(all_passages['1']) == 3:
        passage_with_title = {}
        for id in all_passages.keys():
            passage_with_title[id] = [all_passages[id][0] + " ".join(all_passages[id][2].split(" # ")),
                                      all_passages[id][1], all_passages[id][2]]
    else:
        passage_with_title = all_passages

    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    psgs_doc_hits = validate(
        passage_with_title,
        question_answers,
        top_ids_and_scores,
        args.validation_workers,
        args.match,
    )

    if args.print_doc_hits:
        top_doc_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), 500)
        new_top_ids_and_scores = []
        for example in top_doc_ids_and_scores:
            top_doc_ids = []
            top_doc_scores = []
            all_top_doc = []
            for ids, scores in zip(example[0], list(example[1])):
                title = id_title[ids]
                if title not in all_top_doc:
                    top_doc_ids.append(ids)
                    top_doc_scores.append(scores)
                    all_top_doc.append(title)
            new_top_ids_and_scores.append([top_doc_ids[0:100], top_doc_scores[0:100]])

        docs_doc_hits = validate_doc(
            passage_with_title,
            id_title,
            title_text,
            question_answers,
            new_top_ids_and_scores,
            args.validation_workers,
            args.match,
        )

    if args.out_file:
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            psgs_doc_hits,
            args.out_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--retrieval_doc_file",
        type=str,
        default=None,
        help="the retrieved documents with the title ",
    )
    parser.add_argument(
        "--topk_doc",
        type=int,
        default=None,
        help="the number of topk doc ",
    )
    parser.add_argument(
        "--topk_psgs",
        type=int,
        default=None,
        help="the number of topk doc ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--lamd",
        type=float,
        default=None,
        help="lambda: the coefficience between psg score and doc score ",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--hnsw_index",
        action="store_true",
        help="If enabled, use inference time efficient HNSW index",
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index"
    )
    parser.add_argument(
        "--print_doc_hits", action="store_true", help="If enabled, save index"
    )

    args = parser.parse_args()

    assert (
        args.model_file
    ), "Please specify --model_file checkpoint to init model weights"

    setup_args_gpu(args)
    print_args(args)
    main(args)
