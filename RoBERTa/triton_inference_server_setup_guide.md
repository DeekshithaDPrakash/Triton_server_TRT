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
2. Make triton inference server docker container
   ```python3
   sudo docker run -it --net host --shm-size=4g --name trtis_roberta_ens --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=0"' -v /local_directory_to_mount:/opt/tritonserver/TensorRT_RoBERTa  nvcr.io/nvidia/tritonserver:24.10-py3
   ```
   
   ###### Since this container doesn't come with transformers package pre-installed, we need to install it

   ```python3
   pip install transformers==4.43.0 scipy
   ```

4. Generate config.pbtxt in the below structure:
```
model_repo/
├── ensemble/
│   ├── config.pbtxt
│   ├── 1/  # This folder can be empty but must exist
├── preprocessor/
│   ├── config.pbtxt
│   ├── 1/
├── roberta_classifier/
│   ├── config.pbtxt
│   ├── 1/
├── postprocessor/
│   ├── config.pbtxt
│   ├── 1/
```
##### Use fill_template.py to generate config.pbtxt for preprocess, classifier and post process
```python
python fill_template.py -o model_repo/preprocessor/config.pbtxt \
-n preprocessor -b 8 --backend python \
-i "raw_text:TYPE_STRING:-1" \
-u "input_ids:TYPE_INT32:128;attention_mask:TYPE_INT32:128" \
-p "tokenizer_dir:tokenizer"
```
```python
python3 fill_template.py -o model_repo/roberta_classifier/config.pbtxt \
-n roberta_classifier -b 8 --backend tensorrt_plan --platform tensorrt_plan \
-i "input_ids:TYPE_INT32:128;attention_mask:TYPE_INT32:128" \
-u "logits:TYPE_FP32:3"
```
```python
python3 fill_template.py -o model_repo/postprocessor/config.pbtxt \
-n postprocessor -b 8 --backend python \
-i "logits:TYPE_FP32:-1" \
-u "probabilities:TYPE_FP32:-1,3;predicted_class:TYPE_INT64:-1"
```
##### Use fill_template_ensemble.py to generate config.pbtxt for ensemble
```python
python3 fill_template_ensemble.py \
  -o model_repo/ensemble/config.pbtxt \
  -n ensemble \
  -b 8 \
  -i "raw_text:TYPE_STRING:-1" \
  -u "predicted_class:TYPE_INT64:3;probabilities:TYPE_FP32:-1,3" \
  -e "preprocessor:-1:raw_text=raw_text:input_ids=input_ids,attention_mask=attention_mask;\
roberta_classifier:-1:input_ids=input_ids,attention_mask=attention_mask:logits=logits;\
postprocessor:-1:logits=logits:predicted_class=predicted_class,probabilities=probabilities"
```

##### Add model.py file inside preprocess and post process's "1" Folder
##### Add model.plan inside classifier's "1" Folder

3. Launch the triton server

   ```python
   python3 launch_triton_server.py --model_repo=model_repo --http_port 8040 --grpc_port 8041 --metrics_port 8042
   ```
  ##### curl single input:
   ```python
   curl -X POST localhost:8000/v2/models/ensemble/infer -d '{
     "inputs": [
       {
         "name": "raw_text",
         "shape": [1, 1],
         "datatype": "BYTES",
         "data": [
           "what is your good name?"
         ]
       }
     ],
     "outputs": [
       { "name": "probabilities" },
       { "name": "predicted_class" }
     ]
   }'
```
##### curl multiple inputs
```python
curl -X POST localhost:8000/v2/models/ensemble/infer -d '{
  "inputs": [
    {
      "name": "raw_text",
      "shape": [3, 1],
      "datatype": "BYTES",
      "data": [
        "what is your name",
        "this is so good",
        "He was so scared"
      ]
    }
  ],
  "outputs": [
    { "name": "probabilities" },
    { "name": "predicted_class" }
  ]
}'
```
