import os
import torch
import folder_paths
from huggingface_hub import snapshot_download

from .conf import sana_conf, sana_res
from .loader import load_sana
from nodes import EmptyLatentImage

if not "sana" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("sana", os.path.join(folder_paths.models_dir, "Sana"))


dtypes = [
	"auto",
	"FP32",
	"FP16",
	"BF16"
]

class SanaCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (
					[
						"Efficient-Large-Model/Sana_1600M_1024px_MultiLing",
						"Efficient-Large-Model/Sana_1600M_512px_MultiLing",
						"Efficient-Large-Model/Sana_1600M_1024px_BF16",
						"Efficient-Large-Model/Sana_1600M_1024px",
						"Efficient-Large-Model/Sana_1600M_2Kpx_BF16",
						"Efficient-Large-Model/Sana_1600M_512px",
						"Efficient-Large-Model/Sana_600M_1024px",
						"Efficient-Large-Model/Sana_600M_512px",
					] + folder_paths.get_filename_list("checkpoints"),
				),
				"model": (list(sana_conf.keys()), {"default":"SanaMS_1600M_P1_D20"}),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Checkpoint Loader"

	def load_checkpoint(self, ckpt_name, model):
		if ckpt_name == "Efficient-Large-Model/Sana_1600M_1024px_MultiLing":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-1024px-multilingual")
			model_conf = sana_conf['SanaMS_1600M_P1_D20']
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_1600M_1024px_MultiLing.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_1024px_MultiLing.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_1600M_512px_MultiLing":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-512px-multilingual")
			model_conf = sana_conf['SanaMS_1600M_P1_D20']
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_1600M_512px_MultiLing.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_512px_MultiLing.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_1600M_1024px_BF16":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-1024px-bf16")
			model_conf = sana_conf['SanaMS_1600M_P1_D20']
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_1600M_1024px_BF16.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_1024px_BF16.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_1600M_1024px":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-1024px")
			model_conf = sana_conf['SanaMS_1600M_P1_D20']
			if not os.path.exists(os.path.join(ckpt_path, 'checkpoints/Sana_1600M_1024px.pth')):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_1024px.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_1600M_2Kpx_BF16":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-2kpx-bf16")
			model_conf = sana_conf['SanaMS_1600M_P1_D20_2K']
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_1600M_2Kpx_BF16.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_2Kpx_BF16.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_1600M_512px":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-1600m-512px")
			model_conf = sana_conf["SanaMS_1600M_P1_D20"]
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_1600M_512px.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_1600M_512px.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_600M_1024px":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-600m-1024px")
			model_conf = sana_conf["SanaMS_600M_P1_D28"]
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_600M_1024px.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_600M_1024px.pth"
		elif ckpt_name == "Efficient-Large-Model/Sana_600M_512px":
			ckpt_path = os.path.join(folder_paths.models_dir, "sana", "models--sana--sana-600m-512px")
			model_conf = sana_conf["SanaMS_600M_P1_D28"]
			if not os.path.exists(os.path.join(ckpt_path, "checkpoints/Sana_600M_512px.pth")):
				snapshot_download(ckpt_name, local_dir=ckpt_path)
			ckpt_path = f"{ckpt_path}/checkpoints/Sana_600M_512px.pth"
		else:
			ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
			model_conf = sana_conf[model]

		model = load_sana(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (model,)


class EmptySanaLatentImage(EmptyLatentImage):
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Empty Sana Latent Image"

	def generate(self, width, height, batch_size=1):
		latent = torch.zeros([batch_size, 32, height // 32, width // 32], device=self.device)
		return ({"samples":latent}, )


class SanaResolutionSelect():
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (list(sana_res.keys()),),
				"ratio": (list(sana_res["1024px"].keys()),{"default":"1.00"}),
			}
		}
	RETURN_TYPES = ("INT","INT")
	RETURN_NAMES = ("width","height")
	FUNCTION = "get_res"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Resolution Select"

	def get_res(self, model, ratio):
		width, height = sana_res[model][ratio]
		return (width, height)


class SanaResolutionCond:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"cond": ("CONDITIONING", ),
				"width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
				"height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("cond",)
	FUNCTION = "add_cond"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Resolution Conditioning"
	
	def add_cond(self, cond, width, height):
		for c in range(len(cond)):
			cond[c][1].update({
				"img_hw": [[height, width]],
				"aspect_ratio": [[height/width]],
			})
		return (cond,)


class SanaTextEncode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"text": ("STRING", {"multiline": True}),
				"GEMMA": ("GEMMA",),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	FUNCTION = "encode"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Text Encode"

	def encode(self, text, GEMMA=None):
		tokenizer = GEMMA["tokenizer"]
		text_encoder = GEMMA["text_encoder"]
		
		with torch.no_grad():
			chi_prompt = "\n".join(preset_te_prompt)
			full_prompt = chi_prompt + text
			num_chi_tokens = len(tokenizer.encode(chi_prompt))
			max_length = num_chi_tokens + 300 - 2
			
			tokens = tokenizer(
				[full_prompt],
				max_length=max_length,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			).to(text_encoder.device)
			
			select_idx = [0] + list(range(-300 + 1, 0))
			embs = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
			emb_masks = tokens.attention_mask[:, select_idx]
		embs = embs * emb_masks.unsqueeze(-1)
			
		return ([[embs, {}]], )

preset_te_prompt = [
	'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
	'- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
	'- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
	'Here are examples of how to transform or refine prompts:',
	'- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
	'- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
	'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
	'User Prompt: '
]

NODE_CLASS_MAPPINGS = {
	"SanaCheckpointLoader" : SanaCheckpointLoader,
	"SanaResolutionSelect" : SanaResolutionSelect,
	"SanaTextEncode" : SanaTextEncode,
	"SanaResolutionCond" : SanaResolutionCond,
	"EmptySanaLatentImage": EmptySanaLatentImage,
}
