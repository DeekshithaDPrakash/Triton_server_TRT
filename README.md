This repo consists of guidelines to use Triton inference server using TensorRT/TensorRT-LLM


Note:
1. For Solar models following Llama2 helps 
2. In case "rope_scaling" error pops up, upgrade the transformers to >=4.43 or 4.45.1
3. For multi-gpu, check https://github.com/DeekshithaDPrakash/Triton_server_TRT/blob/eb48ab9350c5d9f0440b359b8ffc7436ce9da391/Llama3/tritonserver_guide_llama3.1_multigpu.md
   It's necessary to use tp_size during convert_checkpoint which creates 2 rank safetensors checkpoints
   and world_size during lauch_triton_server.
   
