#### RoBERTa or any BERT models as of now doesn't support TensorRT-LLM.

#### So, using TensorRT and Triton servers separately is the only way.

### How to make use of triton server for inferencing BERT based models in tensorrt format?

#### Steps:

1. Convert Pytorch saved model (.pt) into TensorRT format (model.plan)
   
   This is 2-step process:
   
   1. Convert .pt into .onnx format using pytorch2onnx.py
      
   3. Convert .onnx into model.plan
      For this we need TensorRT library, which can be installed via docker image
      ```python
      docker run -it --net host --shm-size=4g --name trt_roberta --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=0"' -v     /local_directory_to_mount:/workspace/TensorRT_RoBERTa  nvcr.io/nvidia/tensorrt:24.10-py3
      ```
      Next, run the trtexec command as follows for batched inference:
      ```python
      trtexec --onnx=roberta_task_classifier.onnx \
        --saveEngine=model.plan \
        --minShapes=input_ids:1x128,attention_mask:1x128 \
        --optShapes=input_ids:4x256,attention_mask:4x256 \
        --maxShapes=input_ids:8x256,attention_mask:8x256 \
        --fp16
      ```
      
