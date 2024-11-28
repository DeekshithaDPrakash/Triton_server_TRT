import os
import torch
from torch import nn
from transformers import RobertaModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = 'klue/roberta-small'
roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=768)

# Define the RobertaClassifier model as per your architecture
class RobertaClassifier(nn.Module):
    def __init__(self, base_model, num_classes, dr_rate, hidden_size):
        super().__init__()
        self.dr_rate = dr_rate
        self.roberta = base_model.roberta
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.roberta_classifier = base_model.classifier
        self.dense = nn.Linear(hidden_size, 384)
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, input_ids, attention_mask):
        h_0 = torch.zeros(1, input_ids.size(0), hidden_size).to(input_ids.device)
        c_0 = torch.zeros(1, input_ids.size(0), hidden_size).to(input_ids.device)
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
        out, _ = self.lstm(out, (h_0, c_0))
        out = self.roberta_classifier(out)
        out = self.dense(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

# Load model
best_model = torch.load('roberta-small_task.pt')
pretrained_dict = best_model['state_dict']


num_classes = 3  # Adjust as needed
dr_rate = 0.5
hidden_size = 768

model = RobertaClassifier(roberta_model, num_classes, dr_rate, hidden_size)
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()  # Ensure the model is in evaluation mode
model.to(device)

# Create dummy input for ONNX export
dummy_input_ids = torch.ones((1, 256), dtype=torch.int32).to(device)  # Cast to INT32
dummy_attention_mask = torch.ones((1, 256), dtype=torch.int32).to(device)  # Cast to INT32

# Export to ONNX
onnx_path = "roberta_task_classifier.onnx"
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=13
)
print(f"Model exported to {onnx_path}")
