# -*-coding:utf-8-*-
import os
from pathlib import Path
import lucene
import argparse
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (IndexOptions, IndexWriter, IndexWriterConfig)
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
import sys
import csv

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_index_path",
    type=str,
    help="Output path of the index",
)
parser.add_argument(
    "--input_corpus",
    type=str,
    help="Input path of doc/psg wikipedia corpus",
)
args = parser.parse_args()
env = lucene.initVM()
fsDir = SimpleFSDirectory(Paths.get(args.output_index_path))
writerConfig = IndexWriterConfig(StandardAnalyzer())
writer = IndexWriter(fsDir, writerConfig)

# Define field type
t1 = FieldType()
t1.setStored(True)
t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

t2 = FieldType()
t2.setStored(False)
t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

ctx_file = args.input_corpus
with open(ctx_file) as tsvfile:
    reader = csv.reader(
        tsvfile,
        delimiter="\t",
    )
    # file format: doc_id, doc_text, title
    for row in reader:
        if row[0] != "id":
            doc_id = row[0]
            doc_text = row[1]  # " ".join(eval(row[1]))
            doc_title = row[2]
            doc_title_list = row[3]

            doc_text = doc_text

            doc = Document()
            doc.add(Field('id', doc_id, t1))
            doc.add(Field('title', doc_title, t1))
            doc.add(Field('text', doc_text, t2))
            writer.addDocument(doc)

writer.commit()
writer.close()
print("done!")
