import torch
from safetensors.torch import load_file, save_file
import pdb


# diffusion_model.blocks.9.self_attn.v.lora_down.weight
# lora_unet_blocks_9_self_attn_v.lora_down.weight'
# 把"lora_unet_blocks_"替换为"diffusion_model.blocks." 
# 把 "_self_attn_"替换为".self_attn."
# "_cross_attn_" 替换为".cross_attn."
# "_ffn_" 替换为".ffn."

#
def convert_to_diffusers(weight_dict):
    # 把"lora_unet_blocks_"替换为"diffusion_model.blocks." 
    # 把 "_self_attn_"替换为".self_attn."
    # "_cross_attn_" 替换为".cross_attn."
    # "_ffn_" 替换为".ffn."
    new_weights_sd = {}
    for k, v in weight_dict.items():
        new_k = k.replace("lora_unet_blocks_", "diffusion_model.blocks.")
        new_k = new_k.replace("_self_attn_", ".self_attn.")
        new_k = new_k.replace("_cross_attn_", ".cross_attn.")
        new_k = new_k.replace("_ffn_", ".ffn.")
        new_weights_sd[new_k] = v
        # 如果不是torch.float32，就转换为torch.float32
        if v.dtype != torch.float32:
            v.data = v.data.to(torch.float32)
    return new_weights_sd

prefix = "low"
ref = load_file(f"/data/yaofu/Wan2.2-Lightning/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/{prefix}_noise_model.safetensors")
cur = load_file(f"/data/yaofu/lora_wan/{prefix}_noise_model_ori.safetensors")
cur = convert_to_diffusers(cur)
#保存cur
save_convert = False
if save_convert:
    save_file(cur, f"/data/yaofu/lora_wan/{prefix}_noise_model.safetensors")



# 检查 cur 和 ref 的keys 是不是一样
assert set(cur.keys()) == set(ref.keys())
print(set(cur.keys()) == set(ref.keys()))
# len(up_weight.size()) == 4 检查又没
ref_alph = []
for k, v in ref.items():
    if len(v.size()) != 2:
        ref_alph.append(k)
cur_alph = []
for k, v in cur.items():
    if len(v.size()) != 2:
        cur_alph.append(k)

# 检查 ref_alph 和 cur_alph 是不是一样
assert set(ref_alph) == set(cur_alph)
print(set(ref_alph) == set(cur_alph))
# print(cur["diffusion_model.blocks.9.self_attn.v.alpha"])
# pdb.set_trace()
