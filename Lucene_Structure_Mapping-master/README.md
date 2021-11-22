# Doc-level Psg-level retrieval training dataset creation

We build the train/dev dataset based on the BM25 similiarity score using Pyluence.

If you haven't install Pyluence, please follow
this [Pyluence Install](Lucene_Structure_Mapping-master/Pyluence_Install.md)

## 1. Build Lucene Indexer:

First, you need to use Pylucene Indexer to build the Index.

### Doc-level Retriever:

```
python Lucence_Indexer.py \
        --input_corpus {path of "docs_w100_title.tsv"} \
        --output_index_path {index of doc}
```

### Psg-level Retriever:

```
python Lucence_Indexer.py \
        --input_corpus {path of "new_psgs_w100.tsv"} \
        --output_index_path {index of psg}
```

## 2. Build Lucene Searcher:

Second, use Pylucene Searcher to Search the relative Doc/Psg and set them to positive (has the answer) or negative (
doesn't have the answer).

### Doc-level Retriever (need to process for both train and dev):

```
python Lucence_Searchdoc.py \
        --input_doc_path {path of "docs_w100_title.tsv"} \
        --input_index_path {index of doc} \
        --query_data_path {path of "nq_train.csv" or "nq_dev.csv"} \
        --output_BM25_dataset_path {output_BM25_path_of_train/dev e.g. nq_dev_BM25_doc.json}
```

The output is the retrieved documents for each question based on the BM25 ranking.

### Psg-level Retriever (need to process for both train and dev):

```
python Lucence_Searcher.py \
        --input_psg_path {path of "new_psgs_w100.tsv"} \
        --input_index_path {index of psg} \
        --query_data_path {path of "nq_train.csv" or "nq_dev.csv"} \
        --output_BM25_dataset_path {output_BM25_path_of_train/dev e.g. nq_dev_BM25_psg.json}
```

The output is retrieved positive and hard_negative passages (positive: contains answer, negative: without answer) for
each question based on the BM25 ranking.

Note: We use multiprocess to decrease the time cost. We set number of worker as 50. You can set this number either to
bigger or smaller based on the performance of our machine.

## 3. Generate the Final Retriever Train/Dev Dataset:

After the ranking using BM25, process the following command to get the final train/dev dataset.

### Generate the doc-level Retriever Dataset (need to process for both train and dev):

```
python build_docs_example.py \
        --input_doc_path {path of "new_psgs_w100.tsv"} \
        --gold_data_path {gold_passages_info_path of "nq_train.json" or "nq_dev.json"} \
        --input_BM25_dataset_path {Input_BM25_path_of_train/dev e.g. nq_dev_BM25_doc.json} \ 
        --output_final_dataset_path {output_path}   
```

The output is positive and negative documents for each question.

### Generate the Psg-level Retriever Dataset (need to process for both train and dev):

```
python build_psgs_example.py \
        --input_psg_path {path of "new_psgs_w100.tsv"} \
        --gold_data_path {gold_passages_info_path of "nq_train.json" or "nq_dev.json"} \
        --input_BM25_dataset_path {Input_BM25_path_of_train/dev e.g. nq_dev_BM25_psg.json} \
        --output_final_dataset_path {output_path}  
```

The output is positive and negative passages (in article, hard negative negative) for each question. 