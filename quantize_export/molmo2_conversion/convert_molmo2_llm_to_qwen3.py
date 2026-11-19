"""Molmo2-4B의 언어 모델 부분을 HF Qwen3ForCausalLM 체크포인트로 변환.

- att_proj(fused q|k|v) -> q_proj / k_proj / v_proj
- ff_proj(fused up|gate, Molmo2MLP: x, gate = chunk(2)) -> up_proj / gate_proj
- wte.embedding(151936) -> embed_tokens, lm_head와 동일하므로 tie_word_embeddings=True
- wte.new_embedding(128, 영상 특수토큰)은 LLM 어휘에 넣지 않고 molmo2_new_embedding.safetensors로 따로 저장.
  lm_head를 0으로 패딩해 어휘를 늘리면 logit 0인 특수토큰이 생성되는 문제가 있어서,
  영상 블록 임베딩은 시각부가 만들어 LLM에 주입하는 구조로 둔다.
- 토크나이저와 chat template은 Molmo2 것을 그대로 복사

usage: python convert_molmo2_llm_to_qwen3.py <molmo2_hf_dir> <qwen3_config.json> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, qwen_cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
DTYPE = torch.bfloat16

mcfg = json.load(open(src / "config.json"))
tcfg = mcfg["text_config"]
weight_map = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
handles = {f: safe_open(src / f, "pt") for f in set(weight_map.values())}


def get(name):
    return handles[weight_map[name]].get_tensor(name)


n_q, n_kv, hd = tcfg["num_attention_heads"], tcfg["num_key_value_heads"], tcfg["head_dim"]
q_dim, kv_dim = n_q * hd, n_kv * hd
inter = tcfg["intermediate_size"]
vocab = tcfg["vocab_size"]

sd = {}
emb = get("model.transformer.wte.embedding")
assert torch.equal(get("lm_head.weight"), emb), "lm_head와 embedding이 달라 tie할 수 없음"
sd["model.embed_tokens.weight"] = emb
sd["model.norm.weight"] = get("model.transformer.ln_f.weight")
save_file({"new_embedding": get("model.transformer.wte.new_embedding").to(torch.bfloat16).contiguous()},
          out / "molmo2_new_embedding.safetensors")

for i in range(tcfg["num_hidden_layers"]):
    p, o = f"model.transformer.blocks.{i}", f"model.layers.{i}"
    qkv = get(f"{p}.self_attn.att_proj.weight")
    assert qkv.shape[0] == q_dim + 2 * kv_dim
    sd[f"{o}.self_attn.q_proj.weight"] = qkv[:q_dim]
    sd[f"{o}.self_attn.k_proj.weight"] = qkv[q_dim:q_dim + kv_dim]
    sd[f"{o}.self_attn.v_proj.weight"] = qkv[q_dim + kv_dim:]
    sd[f"{o}.self_attn.o_proj.weight"] = get(f"{p}.self_attn.attn_out.weight")
    sd[f"{o}.self_attn.q_norm.weight"] = get(f"{p}.self_attn.q_norm.weight")
    sd[f"{o}.self_attn.k_norm.weight"] = get(f"{p}.self_attn.k_norm.weight")
    ff = get(f"{p}.mlp.ff_proj.weight")
    assert ff.shape[0] == 2 * inter
    sd[f"{o}.mlp.up_proj.weight"] = ff[:inter]
    sd[f"{o}.mlp.gate_proj.weight"] = ff[inter:]
    sd[f"{o}.mlp.down_proj.weight"] = get(f"{p}.mlp.ff_out.weight")
    sd[f"{o}.input_layernorm.weight"] = get(f"{p}.attn_norm.weight")
    sd[f"{o}.post_attention_layernorm.weight"] = get(f"{p}.ff_norm.weight")

sd = {k: v.to(DTYPE).contiguous() for k, v in sd.items()}
save_file(sd, out / "model.safetensors", metadata={"format": "pt"})

cfg = json.load(open(qwen_cfg_path))
cfg.update(
    vocab_size=vocab, hidden_size=tcfg["hidden_size"], intermediate_size=inter,
    num_hidden_layers=tcfg["num_hidden_layers"], num_attention_heads=n_q, num_key_value_heads=n_kv,
    head_dim=hd, rope_theta=tcfg["rope_theta"], rms_norm_eps=tcfg["layer_norm_eps"],
    max_position_embeddings=tcfg["max_position_embeddings"], tie_word_embeddings=True,
    torch_dtype="bfloat16", bos_token_id=mcfg.get("bos_token_id", cfg.get("bos_token_id")),
    eos_token_id=mcfg.get("eos_token_id", cfg.get("eos_token_id")),
)
cfg.pop("dtype", None)
cfg["rope_scaling"] = None
json.dump(cfg, open(out / "config.json", "w"), indent=2)

for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
          "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"]:
    if (src / f).exists():
        shutil.copy(src / f, out / f)
print(f"saved {len(sd)} tensors, vocab={vocab} -> {out}")
