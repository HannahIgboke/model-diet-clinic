"""Build tiny random models on disk, so tests run offline in seconds."""

from __future__ import annotations

import os

import torch

TEXT = os.path.join(os.path.dirname(__file__), "..", "diet", "data", "clinic_text.txt")


def make_tokenizer(chat: bool):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    with open(TEXT, encoding="utf-8") as f:
        tk.train_from_iterator([f.read()], trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    if chat:
        tok.chat_template = (
            "{% for m in messages %}<s>{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
    return tok


def make_model(kind: str, path: str, dtype=torch.bfloat16, layers: int = 6, chat: bool = False) -> str:
    from transformers import (AutoModelForCausalLM, GPT2Config, LlamaConfig, Qwen2Config)

    tok = make_tokenizer(chat)
    v = len(tok)
    if kind == "llama":
        cfg = LlamaConfig(vocab_size=v, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    elif kind == "qwen2":
        cfg = Qwen2Config(vocab_size=v, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                          tie_word_embeddings=True)
    elif kind == "gpt2":
        cfg = GPT2Config(vocab_size=v, n_embd=64, n_layer=layers, n_head=4, n_positions=512)
    else:
        raise ValueError(kind)
    cfg.bos_token_id, cfg.eos_token_id = tok.bos_token_id, tok.eos_token_id
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(dtype)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    return path
