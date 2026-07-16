# Multimodal-AI-System-for-Dermatology-with-ABCDE-Explainability-and-VQA
## Setup
Download the base VQA model:
```python
from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
```