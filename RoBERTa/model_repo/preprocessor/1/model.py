from transformers import AutoTokenizer
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """
        Triton entry point for initialization.
        Args:
            args (dict): Key-value parameters from Triton config.pbtxt.
        """
        self.tokenizer_dir = "tokenizer"  # Construct tokenizer path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)  # Load tokenizer
        print(f"Tokenizer initialized from directory: {self.tokenizer_dir}")

    def execute(self, requests):
        """
        Triton entry point for preprocessing.
        Args:
            requests (list of pb_utils.InferenceRequest): Batch of requests from Triton.
        Returns:
            list of pb_utils.InferenceResponse: Processed `input_ids` and `attention_mask`.
        """
        responses = []

        for request in requests:
            # Extract the raw_text input tensor
            raw_texts = pb_utils.get_input_tensor_by_name(request, "raw_text").as_numpy()

            # Decode byte strings to UTF-8
            raw_texts = [raw_text.decode("utf-8") for raw_text in raw_texts.flatten()]

            # Tokenize the input text
            tokens = self.tokenizer(
                raw_texts,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            # Prepare `input_ids` and `attention_mask`
            input_ids = tokens["input_ids"].astype(np.int32)
            attention_mask = tokens["attention_mask"].astype(np.int32)

            # Create output tensors
            input_ids_tensor = pb_utils.Tensor("input_ids", input_ids)
            attention_mask_tensor = pb_utils.Tensor("attention_mask", attention_mask)

            # Create Triton inference response
            responses.append(pb_utils.InferenceResponse(output_tensors=[input_ids_tensor, attention_mask_tensor]))

        return responses
