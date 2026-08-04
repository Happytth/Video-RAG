import sys
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig
from PIL import Image

# Global patch for transformers 4.49+ compatibility with Florence-2 remote code
setattr(PretrainedConfig, "forced_bos_token_id", None)
setattr(PretrainedConfig, "forced_eos_token_id", None)

print("Testing Florence-2 imports with PretrainedConfig patch...")
try:
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    print("Florence-2 loaded successfully!")
except Exception as e:
    print("Florence-2 loading error:", e)
