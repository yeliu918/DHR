# DHR: Dense Hierarchical Retrieval for Open-Domain Question Answering

Ye Liu, Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Philip S.
Yu. [Dense Hierarchical Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2110.15439). If you find
this work useful, please cite our paper.

Dense Hierarchical Retrieval (DHR) is a hierarchical framework which can generate accurate dense representations of
passages by utilizing both macroscopic semantics in the document and microscopic semantics specific to each passage.
Specifically, a document-level retriever first identifies relevant documents, among which relevant passages are then
retrieved by a passage-level retriever. The ranking of the retrieved passages will be further calibrated by examining
the document-level relevance. In addition, hierarchical title structure and two negative sampling strategies (i.e., in
document negatives) are investigated.

Our code is built based on [DPR](https://github.com/facebookresearch/DPR). Different from the original DPR, we process
our own

1. Structural document.
2. Dense hierarchical retrieval framework.
3. New passage in-section split.

Compared results between DPR and our proposed DHR on Top-k passages retrieval accuracy on NQ test set.

| Top-k passages | Original DPR NQ model | DHR model | DPR model (2-iter) | DHR model (2-iter) |
| :------------- |:-------------:| :-----:| :-------:| :-------:|
| 1      | 45.87 | 55.37| 52.47 |56.01 |
| 5      | 68.14  |  75.40  |   72.24 |76.73 |
| 20     | 79.97   |  85.07 |    81.33 |85.35 |
| 100    | 85.87 |  89.92   |    87.29 |90.39 |

In the beginning, a bi-encoder model is trained from scratch using the negatives provide in the dataset. The 2-iter
means that uses hard negatives produced by the previous trained model. A bi-encoder model will be trained from scratch
using both the negatives in the dataset and this new produced hard negatives.

## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```
pip install .
```

## Structure Document and New Passage Split

We provide the detail of producing structural document in [Structural Doc](wikiextractor-structure-master/README.md)
The new doc-level and psg-level training dataset construction under the new process wikipedia
in [Training Dataset](Lucene_Structure_Mapping-master/README.md)

You can also directly use our processed [Wikipedia Corpus and Processed Training Dataset](https://drive.google.com/drive/folders/174nKOlF-pR9R0GNXxil3aLK-wBI10_TH?usp=sharing).

## Retriever Data formats

The default data format of the DHR document-level and passage-level retriever training data are in JSON. For
document-level retriever, it contains positive passages ("positive_ctxs") and negative passages ("hard_negative_ctxs")
and some additional information. For passage-level retriever, it contains positive passages ("positive_ctxs") and
negative passages ("hard_negative_ctxs", in-document negative "hard_sec_ctxs") and some additional information. In each
passage, it contains title, title_list and text.

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"title_list": "...",
		"text": "...."
	}],
	"hard_negative_ctxs": [“…”],
	“hard_sec_ctxs": ["..."],  # contains this in Psg-level training data
  },
  ...
]
```

## Train Document-level retrieval:

Retriever training quality depends on its effective batch size. The one reported in the paper used 8 x A100 GPUs.
Execution of the following commands will parallel run all GPUs on one machine:

```
python train_doc_dense_encoder.py \
         --encoder_model_type hf_bert \
         --pretrained_model_cfg bert-base-uncased \
         --train_file Dataset/NQ/nq-train_doc.json \
         --dev_file Dataset/NQ/nq-dev_doc.json \
         --output_dir Output_dhr_d \
         --max_grad_norm 2.0 \
         --warmup_steps 1237 \
         --batch_size 128 \
         --do_lower_case \
         --num_train_epochs 45 \
         --dev_batch_size 256 \
         --val_av_rank_start_epoch 30 \
         --fp16 \
         --hard_negatives 1 \
         --dhr_negatives 0 \
         --contain_title_list \
         --insert_title_list
```

### Train Passage-level retrieval:

```
python train_psg_dense_encoder.py \
         --encoder_model_type hf_bert \
         --pretrained_model_cfg bert-base-uncased \
         --train_file Dataset/NQ/nq-train-in_article.json \
         --dev_file Dataset/NQ/nq-dev-in_article.json \
         --output_dir Output_dhr_p \
         --max_grad_norm 2.0 \
         --warmup_steps 1237 \
         --batch_size 128 \
         --do_lower_case \
         --num_train_epochs 45 \
         --dev_batch_size 256 \
         --val_av_rank_start_epoch 30 \
         --fp16 \
         --hard_negatives 1 \ 
         --dhr_negatives 0 \
         --insec_negatives 1 \
         --contain_title_list
```

DHR uses HuggingFace BERT-base as the encoder by default. Other ready options include Fairseq's ROBERTA (
--encoder_model_type fairseq_roberta), Pytext BERT (--encoder_model_type pytext_bert) models.

## Generate Document-level Embedding:

Generating representation takes a lot of time if you process it on one GPU. It is better to use multiple available GPU
servers by running in parallel. We split the entire document corpus to num_gpu blocks and process each of them on the
given gpu_id. To generate the entire document embeddings, you need to follow the following two steps.

Firstly, modify the following cmd excuation in the generate_doc_multiple to your setting(change-bestepoch)

```
'CUDA_LAUNCH_BLOCKING=%s python generate_doc_dense_embeddings.py ' \
'--gpu_id %s --shard_id %s --num_shards %s ' \
'--model_file {Output_dhr_d/dhr_biencoder-bestepoch} ' \
'--ctx_file Dataset/Wiki_Split/docs_w100_title.tsv ' \
'--out_file Output_dhr_d/Infere_retriever_doc/Wiki ' \
'--batch_size 200 --sequence_length 512 --do_lower_case' % (idx, idx, idx, MAX_SHARE)
```

Secondly, process the following command to generate the document embedding.

```
python generate_doc_multiple.py
```

## Generate Passage-level Embedding:

Similiarly, process the following steps to generate the entire passage embeddings.

Modify the following cmd excuate in the generate_psg_multiple to your setting

```
'CUDA_LAUNCH_BLOCKING=%s python generate_psg_dense_embeddings.py ' \
'--gpu_id %s --shard_id %s --num_shards %s ' \
'--model_file {Output_dhr_p/dhr_biencoder-bestepoch} ' \
'--ctx_file Dataset/Wiki_Split/new_psgs_w100.tsv ' \
'--out_file Output_dhr_p/Infere_retriever_psg/Wiki ' \
'--batch_size 200 --sequence_length 350 --do_lower_case' % (idx, idx, idx, MAX_SHARE)
```

Secondly, process the following command to generate the passage embedding.

```
python generate_psg_multiple.py
```

Note: We use same number of GPUs number and shard number since we get it from 8 * 32G GPU. However, if you don't have
enough GPUs, you can set large shard number and small GPUs number. For example, num_shards=50 on 2 GPUs.

## Retrieve the relative passage:

In the inference stage, we first use the document-level retriever (DHR-D) to extract the Top-k1 relative documents of
the given question. Secondly, the passage-level retriever (DHR-P) will retrieve the Top-k2 relative passages from the
first-step Top-k1 retrieved documents.

Stage one, retrieve the Top-k1 relative documents from the entire document corpus.

```
python dense_doc_retriever.py \
        --model_file {Output_dhr_d/dhr_biencoder-bestepoch} \
        --ctx_file Dataset/Wiki_Split/new_psgs_w100.tsv \
        --docs_file Dataset/Wiki_Split/docs_w100_title.tsv
        --qa_file Dataset/qas/nq-test.csv \
        --encoded_ctx_file Output_dhr_d/Infere_retriever_doc/Wiki_*.pkl \
        --out_file Output_dhr_d/retrieval_output_doc \
        --n-docs 100
```

Stage two, retrieve the Top-k2 relative passages from the retrieved Top-k1 documents with the reranked passage score as
Sim(Q, p) + \lambda * Sim(Q, d_{p}) (simiarity between question and passage plus \lambda times the simiarity between
question and the document that passage belonging to). \lambda is the coefficence and usually \lambda = 1 can provide a
good performance.

```
python dense_psg_retriever_given_doc.py \
        --model_file {Output_dhr_p/dhr_biencoder-bestepoch} \
        --ctx_file Dataset/Wiki_Split/new_psgs_w100.tsv \
        --qa_file Dataset/qas/nq-test.csv \
        --encoded_ctx_file Output_dhr_p/Infere_retriever_psg/Wiki_*.pkl \
        --retrieval_doc_file Output_dhr_d/retrieval_output_doc \
        --out_file Output_dhr_p/retrieval_output_psg \
        --topk_doc 100 \
        --topk_psgs 100 \
        --lamd 1 
```

Different dataset needs different topk_doc to achieve the best performance on the passage-level retrieval.

## Reader model

we use the same setting that used in DPR, the bert-based extractive reading comprehension model. Please
check [DPR](https://github.com/facebookresearch/DPR) for more detail.

### Data pre-processing:

```
python preprocess_reader_data.py \
        --retriever_results {path to the retriever train set results file} 
        --gold_passages {Json file to the gold passages}
        --do_lower_case 
        --pretrained_model_cfg bert-base-uncased 
        --encoder_model_type hf_bert 
        --out_file {path to output dir}
        --is_train_set
```

### Training

```
python train_reader.py \
        --seed 42 \
        --learning_rate 1e-5 \
        --eval_step 2000 \
        --do_lower_case \
        --eval_top_docs 50 \
        --warmup_steps 0 \
        --sequence_length 300 \
        --batch_size 16 \
        --passages_per_question 24 \
        --num_train_epochs 40 \
        --dev_batch_size 72 \
        --passages_per_question_predict 50 \
        --eval_step 4000 \
        --encoder_model_type hf_bert \
        --pretrained_model_cfg bert-base-uncased \
        --train_file {path to the preprocess data of training file} \
        --dev_file {path to the preprocess data of dev file} \
        --output_dir {output path to the reader checkpoint}
```

### Inference

```
python train_reader.py \
        --prediction_results_file {path to a file to write the results to} \
        --eval_top_docs 10 20 40 50 80 100 \
        --dev_file {path to the retriever test results file to evaluate} \
        --model_file {path to the reader checkpoint} \
        --dev_batch_size 80 \
        --passages_per_question_predict 100 \
        --sequence_length 350
```