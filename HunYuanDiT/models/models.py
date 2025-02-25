import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from .attn_layers import Attention, FlashCrossMHAModified, FlashSelfMHAModified, CrossAttention
from .embedders import TimestepEmbedder, PatchEmbed, timestep_embedding
from .norm_layers import RMSNorm
from .poolers import AttentionPool
from .posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class FP32_SiLU(nn.SiLU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)


class HunYuanDiTBlock(nn.Module):
    """
    A HunYuanDiT block with `add` conditioning.
    """
    def __init__(self,
                 hidden_size,
                 c_emb_size,
                 num_heads,
                 mlp_ratio=4.0,
                 text_states_dim=1024,
                 use_flash_attn=False,
                 qk_norm=False,
                 norm_type="layer",
                 skip=False,
                 ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        if use_flash_attn:
            self.attn1 = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        else:
            self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        if use_flash_attn:
            self.attn2 = FlashCrossMHAModified(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True,
                                               qk_norm=qk_norm)
        else:
            self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True,
                                        qk_norm=qk_norm)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

    def forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = (
            self.norm1(x) + shift_msa, freq_cis_img,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (
            self.norm3(x), text_states, freq_cis_img
        )
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """
    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunYuanDiT(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """
    def __init__(
            self, args,
            input_size=(32, 32),
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            log_fn=print,
            cond_style=True,
            cond_res=True,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = args.learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if args.learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        self.norm = args.norm
        self.cond_res = cond_res
        self.cond_style = cond_style

        use_flash_attn = args.infer_mode == 'fa'
        if use_flash_attn:
            log_fn(f"    Enable Flash Attention.")
        qk_norm = True  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=torch.float32))

        # Attention pooling
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=1024)


        self.extra_in_dim = 0        
        if self.cond_res:
                # Image size and crop size conditions
                self.extra_in_dim += 256 * 6
        if self.cond_style:
                # Here we use a default learned embedder layer for future extension.
                self.style_embedder = nn.Embedding(1, hidden_size)
                self.extra_in_dim += hidden_size

        # Text embedding for `add`
        self.last_size = input_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_in_dim += 1024
        self.extra_embedder = nn.Sequential(
            nn.Linear(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=self.text_states_dim,
                            use_flash_attn=use_flash_attn,
                            qk_norm=qk_norm,
                            norm_type=self.norm,
                            skip=layer > depth // 2,
                            )
            for layer in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels)
        self.unpatchify_channels = self.out_channels

    def forward_raw(self,
                x,
                t,
                encoder_hidden_states=None,
                text_embedding_mask=None,
                encoder_hidden_states_t5=None,
                text_embedding_mask_t5=None,
                image_meta_size=None,
                style=None,
                cos_cis_img=None,
                sin_cis_img=None,
                return_dict=False,
                ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """

        text_states = encoder_hidden_states                     # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5               # 2,256,2048
        text_states_mask = text_embedding_mask.bool()           # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()     # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat([text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1)  # 2,205，1024
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask
        text_states = torch.where(clip_t5_mask.unsqueeze(2), text_states, self.text_embedding_padding.to(text_states))

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        if self.cond_res:
                # Build image meta size tokens
                image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)   # [B * 6, 256]
                # if self.args.use_fp16:
                    # image_meta_size = image_meta_size.half()
        
                image_meta_size = image_meta_size.view(-1, 6 * 256)
                extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        if self.cond_style:
                # Build style tokens
                style_embedding = self.style_embedder(style)
                extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec.to(self.dtype))  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                skip = skips.pop()
                x = block(x, c, text_states, freqs_cis_img, skip)   # (N, L, D)
            else:
                x = block(x, c, text_states, freqs_cis_img)         # (N, L, D)

            if layer < (self.depth // 2 - 1):
                skips.append(x)

        # ========================= Final layer =========================
        x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)                          # (N, out_channels, H, W)

        if return_dict:
            return {'x': x}
        return x
   
    def calc_rope(self, height, width):
        """
        Probably not the best in terms of perf to have this here
        """
        th = height // 8 // self.patch_size
        tw = width // 8 // self.patch_size
        base_size = 512 // 8 // self.patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = get_2d_rotary_pos_embed(self.head_size, *sub_args)
        return rope

    def forward(self, x, timesteps, context, context_mask=None, context_t5=None, context_t5_mask=None, src_size_cond=(1024,1024), **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 77, C) CLIP conditioning
        context_t5: (N, 1, 256, C) MT5 conditioning
        """
        # context_mask = torch.zeros(x.shape[0], 77, device=x.device)
        # context_t5_mask = torch.zeros(x.shape[0], 256, device=x.device)

        # style
        style = torch.as_tensor([0] * (x.shape[0]), device=x.device)

        # image size - todo separate for cond/uncond when batched
        if torch.is_tensor(src_size_cond):
                src_size_cond = (int(src_size_cond[0][0]), int(src_size_cond[0][1]))
        
        image_size = (x.shape[2]//2*16, x.shape[3]//2*16)
        size_cond = list(src_size_cond) + [image_size[1], image_size[0], 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * x.shape[0], device=x.device)

        # RoPE
        rope = self.calc_rope(*image_size)

        # Update x_embedder if image size changed
        if self.last_size != image_size:
                from tqdm import tqdm
                tqdm.write(f"HyDiT: New image size {image_size}")
                self.x_embedder.update_image_size(
                        (image_size[0]//8, image_size[1]//8), 
                )
                self.last_size = image_size

        # Run original forward pass
        out = self.forward_raw(
                x = x.to(self.dtype),
                t = timesteps.to(self.dtype),
                encoder_hidden_states = context.to(self.dtype),
                text_embedding_mask   = context_mask.to(self.dtype),
                encoder_hidden_states_t5 = context_t5.to(self.dtype),
                text_embedding_mask_t5   = context_t5_mask.to(self.dtype),
                image_meta_size = image_meta_size.to(self.dtype),
                style = style,
                cos_cis_img = rope[0],
                sin_cis_img = rope[1],
        )
        
        # return
        out = out.to(torch.float)
        if self.learn_sigma:
                eps, rest = out[:, :self.in_channels], out[:, self.in_channels:]
                return eps
        else:
                return out

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
