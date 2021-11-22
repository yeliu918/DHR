#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dhr.utils.data_utils import Tensorizer
from dhr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
    ],
)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
            self,
            question_model: nn.Module,
            ctx_model: nn.Module,
            fix_q_encoder: bool = False,
            fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
            sub_model: nn.Module,
            ids: T,
            segments: T,
            attn_mask: T,
            fix_encoder: bool = False,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids, segments, attn_mask
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids, segments, attn_mask
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
            self,
            question_ids: T,
            question_segments: T,
            question_attn_mask: T,
            context_ids: T,
            ctx_segments: T,
            ctx_attn_mask: T,
    ) -> Tuple[T, T]:

        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            self.question_model,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            self.ctx_model,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_ctx_encoder,
        )

        return q_pooled_out, ctx_pooled_out

    @classmethod
    def create_biencoder_input(
            cls,
            samples: List,
            tensorizer: Tensorizer,
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            num_hard_from_psgs_negatives: int = 0,
            num_hard_dhr_negatives: int = 0,
            contain_title_list: bool = False,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        all_batch_ctxs = []
        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if sample["positive_ctxs"] == []:
                continue
            hard_sec_ctxs = None
            dhr_negative_ctxs = None

            if shuffle and shuffle_positives:
                positive_ctxs = sample["positive_doc"]
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample["positive_ctxs"][0]

            hard_neg_ctxs = sample["hard_negative_ctxs"][0:40]
            question = normalize_question(sample["question"])
            if "from_psgs_negative_docs" in sample.keys():
                hard_sec_ctxs = sample["from_psgs_negative_ctxs"]
            if "hard_negative_docs" in sample.keys():
                dhr_negative_ctxs = sample["hard_negative_ctxs"][0:40]

            if shuffle:
                # random.shuffle(neg_ctxs)
                if hard_neg_ctxs:
                    random.shuffle(hard_neg_ctxs)
                if hard_sec_ctxs:
                    random.shuffle(hard_sec_ctxs)
                if dhr_negative_ctxs:
                    random.shuffle(dhr_negative_ctxs)

            if hard_sec_ctxs == []:
                hard_sec_ctxs = hard_neg_ctxs
            # neg_ctxs = neg_ctxs[0:num_other_negatives]
            if hard_neg_ctxs:
                hard_neg_ctxs = hard_neg_ctxs[0: num_hard_negatives]
            if hard_sec_ctxs:
                hard_sec_ctxs = hard_sec_ctxs[0: num_hard_from_psgs_negatives]
            if dhr_negative_ctxs:
                dhr_negative_ctxs = dhr_negative_ctxs[0: num_hard_dhr_negatives]

            all_ctxs = [positive_ctx]
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1
            for neg_ctx in [hard_neg_ctxs, hard_sec_ctxs, dhr_negative_ctxs]:
                if neg_ctx:
                    all_ctxs = all_ctxs + neg_ctx
                    hard_negatives_end_idx = hard_negatives_end_idx + len(neg_ctx)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx["text"], title_list=", ".join(ctx["title_list"].split(" # ")) if contain_title_list else None,
                    title=ctx["title"] if insert_title else None,
                    # ", ".join(ctx["title_list"].split(" # ")) if "title_list" in ctx else
                )
                for ctx in all_ctxs
            ]

            all_batch_ctxs.extend(all_ctxs)
            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx,
                )
                ]
            )
            ## question length is short and we set the maximum length as 80 tokens
            question_tensors.append(tensorizer.text_to_tensor(question)[0:80])

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
        )


class BiEncoderNllLoss(object):
    def calc(
            self,
            q_vectors: T,
            ctx_vectors: T,
            positive_idx_per_question: list,
            hard_negatice_idx_per_question: list = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores
