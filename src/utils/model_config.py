"""Single source of truth for turning YAML into model-constructor kwargs.

Training, evaluation and the parameter/FLOP cost table must build a model from
the SAME configuration, or a checkpoint silently fails to load (or worse, loads
into a differently-shaped model and is reported anyway). ``scripts/evaluate.py``
and ``scripts/model_cost.py`` used to hardcode ``hidden_size=128, num_layers=4``
and nothing else, which built GPT at its code default d_model=256 instead of the
128 it trains at. (F7)

Three normalizations live here:

* ``MODEL_CONFIG_ALIAS`` — registry names without a YAML of their own. (F9)
* ``normalize_model_kwargs`` — expose width/depth under BOTH the
  hidden_size/num_layers and d_model/n_layers spellings. (F8)
* ``flatten_nested_model_cfg`` — lift the nested ``attention:`` block (legacy
  spelling ``cross_attention:``) to the flat kwarg names the modules read. (F4)
  The ``film:`` block was dropped with the FiLM modules themselves.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import load_config

# Repo root, so config/*.yaml resolves the same from any working directory.
# evaluate.py / model_cost.py add the repo to sys.path from __file__ but used
# to hand compose_config CWD-relative paths: run from anywhere but the repo
# root, every checkpoint build raised and was swallowed by the caller's
# except-handler, producing a baseline-only results.csv with exit 0.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    """Absolute path for a config file, relative paths taken from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


# Registry names whose hyperparameters live in a differently-named YAML.
# ``mamba2`` IS the wrapper configured by mamba.yaml (mamba.py registers both
# names); ``mamba`` is the back-compat alias for the same model.
MODEL_CONFIG_ALIAS = {
    "mamba2": "mamba",
    "mamba": "mamba",
}

# Models whose _build_model reads d_model/n_layers rather than the
# hidden_size/num_layers names used by base.yaml and the --num-layers flag.
_WIDTH_ALIASES = {"hidden_size": "d_model", "num_layers": "n_layers"}

# Nested YAML blocks -> the flat kwarg names BaseModel / the combo modules read.
_NESTED_KWARGS = {
    # "attention" is the current block name; "cross_attention" is the legacy
    # spelling, kept because the operator it configured is self-attention and
    # old variant files still carry it.
    "attention": {
        "num_heads": "cross_attn_num_heads",
        "d_k": "cross_attn_d_k",
        "d_v": "cross_attn_d_v",
        "dropout": "attn_dropout",
    },
    "cross_attention": {
        "num_heads": "cross_attn_num_heads",
        "d_k": "cross_attn_d_k",
        "d_v": "cross_attn_d_v",
        "dropout": "attn_dropout",
    },
}


def resolve_model_config(model: str, root: str = "config/models") -> str:
    """Path of the YAML that configures ``model`` (see MODEL_CONFIG_ALIAS)."""
    return f"{root}/{MODEL_CONFIG_ALIAS.get(model, model)}.yaml"


def normalize_model_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Make width/depth reachable under BOTH naming conventions.

    Every model class takes ``**kwargs``, so supplying both spellings is safe:
    each class picks up the one it declares and ignores the other. This is what
    lets one ``--num-layers`` / base width apply uniformly across the roster.
    """
    out = dict(model_cfg)
    for src, dst in _WIDTH_ALIASES.items():
        if src in out and dst not in out:
            out[dst] = out[src]
        elif dst in out and src not in out:
            out[src] = out[dst]
    return out


def flatten_nested_model_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lift the ``attention:`` sub-dict to the flat kwarg names modules read."""
    out = dict(model_cfg)
    for block, mapping in _NESTED_KWARGS.items():
        sub = out.pop(block, None)
        if not isinstance(sub, dict):
            continue
        for src, dst in mapping.items():
            if src in sub:
                out.setdefault(dst, sub[src])
    return out


def compose_data_config(
    base: str = "config/base.yaml",
    extra: Sequence[str] = ("config/census.yaml",),
) -> Dict[str, Any]:
    """Compose the files that can carry a ``data:`` block.

    Only ``config/base.yaml`` and the dataset override define ``data:`` — no
    model or variant YAML does — so the data contract is model-independent.
    Evaluation, the cost panel and the tabular aggregate runner used to build
    ``CensusConfig`` from its dataclass defaults instead, which happened to
    equal ``config/census.yaml`` but was not bound to it: changing
    ``input_len`` / ``lag_count`` / ``lag_mode`` there would have moved the
    training window while evaluation silently kept scoring the old one. Same
    failure mode as F7/F10, on the data side.
    """
    cfg = load_config(str(_resolve(base)))
    for path in extra or ():
        resolved = _resolve(path)
        if resolved.exists():
            cfg = load_config(str(resolved), cfg)
    return cfg


def compose_config(
    model: str,
    variant: str,
    base: str = "config/base.yaml",
    extra: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compose base < model < variant < extra, exactly as scripts/train.py does.

    ``extra`` is the tail of the chain (e.g. ``["config/census.yaml"]``), merged
    last so its data/embedding_dims win.
    """
    cfg = load_config(str(_resolve(base)))
    model_yaml = _resolve(resolve_model_config(model))
    if model_yaml.exists():
        cfg = load_config(str(model_yaml), cfg)
    variant_yaml = _resolve(f"config/variants/{variant}.yaml")
    if variant_yaml.exists():
        cfg = load_config(str(variant_yaml), cfg)
    for path in extra or ():
        resolved = _resolve(path)
        if resolved.exists():
            cfg = load_config(str(resolved), cfg)
    return cfg


def model_kwargs_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Model-constructor kwargs from a composed config.

    Drops the keys the caller always passes explicitly (name/variant and the
    pipeline-derived dimensions), applies the nested-block and width/depth
    normalizations, translates ``embedding_dims``, and threads ``input_len``
    (which lives under ``data:``) through for S4ND's Time-axis kernels. (F10)
    """
    model_cfg = dict(cfg.get("model", {}))
    for key in ("name", "variant", "num_states", "num_commodities",
                "num_flows", "num_numeric_features"):
        model_cfg.pop(key, None)

    model_cfg = flatten_nested_model_cfg(model_cfg)
    model_cfg = normalize_model_kwargs(model_cfg)
    model_cfg.setdefault("input_len", cfg.get("data", {}).get("input_len", 36))

    emb = model_cfg.pop("embedding_dims", None)
    if emb:
        model_cfg.setdefault("state_embed_dim", emb.get("state", 8))
        model_cfg.setdefault("comm_embed_dim", emb.get("commodity", 32))
        model_cfg.setdefault("flow_embed_dim", emb.get("flow", 2))
    return model_cfg
