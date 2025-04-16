import torch.nn as nn
from loralib import LoRALayer

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1, bias=True): #r = LoRA rank controls trainable parameters
        super().__init__(in_features, out_features, bias=bias)
        #initialize LoRA attributes
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        

def apply_lora_to_vit(model, target_keywords=["qkv"]):
    """
    Apply LoRA to attention layers of MobileSAM ViT

    Args:
    model: image encoder (TinyViT)
    target_keywords: Keywords to match target attention layers ("qkv")
    """
    #iterate through all submodules
    for name, module in model.named_modules():
        #target layers with keywords "qkv"
        if isinstance(module, nn.Linear) and any(k in name.lower() for k in target_keywords):
            # save original weights
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            #LoRA linear layer 
            lora_layer = LoRALinear(in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1, bias=bias)
            #copy weights from OG linear layer
            lora_layer.weight.data = module.weight.data.clone()
            if bias:
                lora_layer.bias.data = module.bias.data.clone()
            
            #replace OG layer with LoRA layer
            parent = get_parent_module(model, name) #find OG layer
            last_name = name.split(".")[-1] 
            setattr(parent, last_name, lora_layer) #replace

            print(f"LoRA applied to {name}")


def get_parent_module(model, name):
    names = name.split(".")
    module = model
    for n in names[:-1]:
        module = getattr(module, n)
    return module
