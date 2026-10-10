import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os

os.makedirs("models" , exist_ok=True)
model_name = "bert-base-uncased"

class BERTClassifier(nn.Module):
    def __init__(self, model_name,num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size,num_classes)

    def forward(self,input_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:,0,:]
        logits = self.classifier(cls_output)
        return logits

def main():
    print("Loading PyTorch BERT model and Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BERTClassifier(model_name, num_classes=2)
    model.eval()

    sample_text = "This product is absolutely amazing!"
    inputs = tokenizer(sample_text,max_length=16,padding="max_length",truncation=True,return_tensors="pt")

    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]
    onnx_path= "models/bert_classifier.onnx"

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    print(f"Exporting Model to {onnx_path}")
    torch.onnx.export(model,(dummy_input_ids,dummy_attention_mask),onnx_path,export_params=True,opset_version=14,do_constant_folding=True,input_names=["input_ids","attention_mask"],
                      output_names=["logits"],dynamic_axes=dynamic_axes)
    tokenizer.save_pretrained("models/bert_tokenizer")
    print("✅ BERT ONNX Export Completed Successfully!")

if __name__ == "__main__":
    main()

