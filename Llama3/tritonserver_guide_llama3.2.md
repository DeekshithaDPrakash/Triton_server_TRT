1. Make docker image: choose 24.10 as it comes with transformers 4.45.1

```python
docker run -it --net host --shm-size=4g --name your_docker_image_name --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=0"' -v /your_local_repo_with_llama3.2_model_or_finetuned_llama3.2:/opt/tritonserver/TensorRT_LLM nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
```
2. Clone the tesnorrtllm_backend : I chose the same release branch as my docker container version

`cd TensorRT_LLM`   ----> this is the directory/folder we mounted when creating docker image
```python
git clone -b r24.10 https://github.com/triton-inference-server/tensorrtllm_backend.git
```

`cd tensorrtllm_backend`

```python
 #adding tensorrt_llm submodule
git lfs install
git submodule update --init --recursive`
```

3. Define checkpoint file path and others

```python
CONVERT_CHKPT_SCRIPT=/opt/tritonserver/TensorRT_LLM/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
LLAMA_MODEL=/opt/tritonserver/TensorRT_LLM/llama3.2_model
UNIFIED_CKPT_PATH=/opt/tritonserver/TensorRT_LLM/ckpt/llaam32/3b
ENGINE_DIR=/opt/tritonserver/TensorRT_LLM/engines/1-gpu

```
4. Execute convert_checkpoint.py with --use_embedding_sharing tag

```python
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${LLAMA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16 --use_embedding_sharing
``` 

5. Build the engine
```python
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
            --remove_input_padding enable \
            --gpt_attention_plugin float16 \
            --context_fmha enable \
            --gemm_plugin float16 \
            --output_dir ${ENGINE_DIR} \
            --paged_kv_cache enable \
            --max_batch_size 8
``` 

6. Make config.pbtxt files using inflight_batcher_llm

```python
COMMON_DIR=/opt/tritonserver/
cp -r /opt/tritonserver/TensorRT_LLM_KARI/tensorrtllm_backend/tools ${COMMON_DIR}
cp -r /opt/tritonserver/TensorRT_LLM_KARI/tensorrtllm_backend/scripts ${COMMON_DIR}
cp -R /opt/tritonserver/TensorRT_LLM_KARI/tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.

```
```python
python3 ${COMMON_DIR}tools/fill_template.py -i ${COMMON_DIR}inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1

python3 ${COMMON_DIR}tools/fill_template.py -i ${COMMON_DIR}inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1

python3 ${COMMON_DIR}tools/fill_template.py -i ${COMMON_DIR}inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False

python3 ${COMMON_DIR}tools/fill_template.py -i ${COMMON_DIR}inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:64

python3 ${COMMON_DIR}tools/fill_template.py -i ${COMMON_DIR}inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_DIR},max_tokens_in_paged_kv_cache:81920,max_attention_window_size:81920,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```

7. Launch the triton server with desired port numbers
```python
python3 /opt/tritonserver/scripts/launch_triton_server.py --world_size 1 --model_repo=/opt/tritonserver/inflight_batcher_llm --http_port 8010 --grpc_port 8011 --metrics_port 8012
````
