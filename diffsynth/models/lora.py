import torch
# 只保留WanX相关的模型导入
from .wan_video_dit import WanModel


class LoRAFromCivitai:
    def __init__(self):
        self.supported_model_classes = []
        self.lora_prefix = []
        self.renamed_lora_prefix = {}
        self.special_keys = {}

    def convert_state_dict(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        for key in state_dict:
            if ".lora_up" in key:
                return self.convert_state_dict_up_down(state_dict, lora_prefix, alpha)
        return self.convert_state_dict_AB(state_dict, lora_prefix, alpha)

    def convert_state_dict_up_down(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        renamed_lora_prefix = self.renamed_lora_prefix.get(lora_prefix, "")
        state_dict_ = {}
        for key in state_dict:
            if ".lora_up" not in key:
                continue
            if not key.startswith(lora_prefix):
                continue
            weight_up = state_dict[key].to(device="cuda", dtype=torch.float16)
            weight_down = state_dict[key.replace(".lora_up", ".lora_down")].to(device="cuda", dtype=torch.float16)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2).to(torch.float32)
                weight_down = weight_down.squeeze(3).squeeze(2).to(torch.float32)
                lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_up, weight_down)
            target_name = key.split(".")[0].replace(lora_prefix, renamed_lora_prefix).replace("_", ".") + ".weight"
            for special_key in self.special_keys:
                target_name = target_name.replace(special_key, self.special_keys[special_key])
            state_dict_[target_name] = lora_weight.cpu()
        return state_dict_

    def convert_state_dict_AB(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        renamed_lora_prefix = self.renamed_lora_prefix.get(lora_prefix, "")
        state_dict_ = {}
        for key in state_dict:
            if not key.startswith(lora_prefix):
                continue
            weight = state_dict[key].to(device="cuda", dtype=torch.float32)
            target_name = key.replace(lora_prefix, renamed_lora_prefix).replace("_", ".") + ".weight"
            for special_key in self.special_keys:
                target_name = target_name.replace(special_key, self.special_keys[special_key])
            state_dict_[target_name] = alpha * weight.cpu()
        return state_dict_

    def match(self, model, state_dict):
        if not any(isinstance(model, model_cls) for model_cls in self.supported_model_classes):
            return None
        state_dict_model = model.state_dict()
        for lora_prefix in self.lora_prefix:
            for model_resource in ["civitai", "diffusers"]:
                try:
                    state_dict_lora_ = {}
                    for name in state_dict:
                        if name.startswith(lora_prefix):
                            state_dict_lora_[name] = state_dict[name]
                    state_dict_lora_ = self.convert_state_dict(state_dict_lora_, lora_prefix, alpha=1.0)
                    converter_fn = model.__class__.state_dict_converter().from_diffusers \
                        if model_resource == "diffusers" \
                        else model.__class__.state_dict_converter().from_civitai
                    state_dict_lora_ = converter_fn(state_dict_lora_)
                    if isinstance(state_dict_lora_, tuple):
                        state_dict_lora_ = state_dict_lora_[0]
                    if len(state_dict_lora_) == 0:
                        continue
                    for name in state_dict_lora_:
                        if name not in state_dict_model:
                            break
                    else:
                        return lora_prefix, model_resource
                except:
                    pass
        return None

    def load(self, model, state_dict, lora_prefix, alpha=1.0, model_resource="civitai"):
        state_dict_model = model.state_dict()
        state_dict_lora = {}
        for name in state_dict:
            if name.startswith(lora_prefix):
                state_dict_lora[name] = state_dict[name]
        state_dict_lora = self.convert_state_dict(state_dict_lora, lora_prefix, alpha)
        converter_fn = model.__class__.state_dict_converter().from_diffusers \
            if model_resource == "diffusers" \
            else model.__class__.state_dict_converter().from_civitai
        state_dict_lora = converter_fn(state_dict_lora)
        if isinstance(state_dict_lora, tuple):
            state_dict_lora = state_dict_lora[0]
        for name in state_dict_lora:
            state_dict_model[name] += state_dict_lora[name].to(device=state_dict_model[name].device, dtype=state_dict_model[name].dtype)
        print(f"    {len(state_dict_lora)} tensors are updated.")
        model.load_state_dict(state_dict_model)


class GeneralLoRAFromPeft:
    """WanX专用的通用LoRA加载器"""
    def __init__(self):
        self.supported_model_classes = [WanModel]
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            prefix = key.replace(".lora_B.default.weight", "")
            lora_name_dict[prefix + ".weight"] = [
                prefix + ".lora_A.default.weight",
                prefix + ".lora_B.default.weight"
            ]
        return lora_name_dict

    def match(self, model, state_dict):
        if not any(isinstance(model, model_cls) for model_cls in self.supported_model_classes):
            return None
        lora_name_dict = self.get_name_dict(state_dict)
        if len(lora_name_dict) == 0:
            return None
        state_dict_model = model.state_dict()
        model_keys = set(state_dict_model.keys())
        lora_keys = set(lora_name_dict.keys())
        match_ratio = len(model_keys & lora_keys) / len(lora_keys)
        if match_ratio > 0.8:
            return "", ""
        return None

    def fetch_device_and_dtype(self, state_dict_model):
        device, dtype = None, None
        for param in state_dict_model.values():
            device, dtype = param.device, param.dtype
            break
        computation_device, computation_dtype = device, dtype
        if computation_device == torch.device("cpu"):
            if torch.cuda.is_available():
                computation_device = torch.device("cuda")
        if computation_dtype == torch.float8_e4m3fn:
            computation_dtype = torch.float32
        return device, dtype, computation_device, computation_dtype

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        state_dict_model = model.state_dict()
        device, dtype, computation_device, computation_dtype = self.fetch_device_and_dtype(state_dict_model)
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name in lora_name_dict:
            weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
            weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
            weight_patched = weight_model + weight_lora
            state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
        print(f"    {len(lora_name_dict)} tensors are updated.")
        model.load_state_dict(state_dict_model)


class WanLoRAConverter:
    """WanX专用LoRA转换器"""
    def __init__(self):
        pass

    @staticmethod
    def align_to_opensource_format(state_dict, **kwargs):
        state_dict = {"diffusion_model." + name.replace(".default.", "."): param for name, param in state_dict.items()}
        return state_dict
    
    @staticmethod
    def align_to_diffsynth_format(state_dict, **kwargs):
        state_dict = {name.replace("diffusion_model.", "").replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): param for name, param in state_dict.items()}
        return state_dict


def get_lora_loaders():
    """只返回WanX支持的LoRA加载器"""
    return [GeneralLoRAFromPeft()]
