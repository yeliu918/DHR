import os
import time
from multiprocessing import Pool
import torch
from gpuinfo import GPUInfo

MAX_SHARE = torch.cuda.device_count()


def generate_dense_embeddings(idx):
    cmd = 'CUDA_LAUNCH_BLOCKING=%s python generate_doc_dense_embeddings.py ' \
          '--gpu_id %s --shard_id %s --num_shards %s ' \
          '--model_file Output_dhr_d/dhr_biencoder_best ' \
          '--ctx_file Dataset/Wiki_Split/docs_w100_title.tsv ' \
          '--out_file Output_dhr_d/Infere_retriever_doc/Wiki ' \
          '--batch_size 400 --sequence_length 512 --do_lower_case' % (idx, idx, idx, MAX_SHARE)
    print('\n' + cmd)
    os.system(cmd)


def gpu_info(interval=600):
    available_device = MAX_SHARE
    percent, memory = GPUInfo.gpu_usage()
    min_memory = min([memory[i] for i in range(available_device)])
    print("the number of device is", available_device)
    print("the min memory is ", min_memory)
    while min_memory > 35000:
        print("waiting")
        percent, memory = GPUInfo.gpu_usage()
        min_memory = min([memory[i] for i in range(available_device)])
        time.sleep(interval)
    workers_num = available_device
    processes = Pool(
        processes=workers_num,
    )
    processes.map(generate_dense_embeddings, range(available_device))


if __name__ == '__main__':
    gpu_info()
