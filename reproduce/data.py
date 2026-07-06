import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import typing as tp
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.utils import DownloadConfig


log = logging.getLogger(__name__)

DATA_CACHE_VERSION = os.environ.get("CARE_LORA_DATA_CACHE_VERSION", "v1")

_REPRODUCE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROCESSED_DATA_ROOT = os.path.join(_REPRODUCE_DIR, "processed_datasets")
_GSM8K_CACHE = os.path.join(_PROCESSED_DATA_ROOT, "gsm8k_main")
_METAMATHQA_CACHE = os.path.join(_PROCESSED_DATA_ROOT, "metamathqa_configurable")
_OPENCODEINSTRUCT_CACHE = os.path.join(_PROCESSED_DATA_ROOT, "opencodeinstruct_configurable")
_SMOLTALK_SMOL_MAGPIE_ULTRA_CACHE = os.path.join(
    _PROCESSED_DATA_ROOT,
    "smoltalk_smol_magpie_ultra_configurable",
)

_METAMATH_HUB_ID = "meta-math/MetaMathQA"
_NETWORK_ERROR_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "Timeout",
    "OfflineModeIsEnabled",
}


def _is_network_or_offline_error(error: Exception) -> bool:
    name = type(error).__name__
    msg = str(error).lower()
    return (
        name in _NETWORK_ERROR_NAMES
        or "couldn't reach" in msg
        or "could not connect" in msg
        or "connection" in msg
        or "offline" in msg
        or "timed out" in msg
    )


def _env_path(*names: str) -> str:
    for name in names:
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _load_local_train_split(local: str, *, label: str) -> Dataset:
    if not os.path.isdir(local):
        raise FileNotFoundError(f"{label} local dataset path does not exist or is not a directory: {local!r}")
    data = load_from_disk(local)
    if isinstance(data, DatasetDict):
        if "train" not in data:
            raise KeyError(f"{label} local DatasetDict has no train split; keys={list(data.keys())}")
        return data["train"]
    return data


def _load_hub_split(
    *dataset_args,
    split: str,
    label: str,
    local_envs: tp.Sequence[str] = (),
    streaming: bool = False,
):
    local = _env_path(*local_envs)
    if local:
        log.info("Loading %s from local dataset path: %s", label, local)
        return _load_local_train_split(local, label=label)

    try:
        return load_dataset(*dataset_args, split=split, streaming=bool(streaming))
    except Exception as error:
        if not _is_network_or_offline_error(error):
            raise
        log.warning(
            "%s download failed (%s: %s); retrying with local Hugging Face cache.",
            label,
            type(error).__name__,
            error,
        )
        try:
            return load_dataset(
                *dataset_args,
                split=split,
                streaming=bool(streaming),
                download_config=DownloadConfig(local_files_only=True),
            )
        except Exception as cached_error:
            local_hint = ", ".join(local_envs) if local_envs else "a local dataset path"
            raise ConnectionError(
                f"Unable to load {label}: Hugging Face Hub is unavailable and no local cache was found.\n"
                f"Set one of these environment variables to a Dataset.save_to_disk directory: {local_hint}.\n"
                f"Hub error: {error!r}\n"
                f"local_files_only retry: {cached_error!r}"
            ) from cached_error


def _load_dataset_dict(*dataset_args, label: str):
    try:
        return load_dataset(*dataset_args)
    except Exception as error:
        if not _is_network_or_offline_error(error):
            raise
        log.warning(
            "%s download failed (%s: %s); retrying with local Hugging Face cache.",
            label,
            type(error).__name__,
            error,
        )
        return load_dataset(*dataset_args, download_config=DownloadConfig(local_files_only=True))


def _load_metamath_qa_train_split():
    """Load the MetaMathQA train split with local/offline fallbacks."""
    return _load_hub_split(
        _METAMATH_HUB_ID,
        split="train",
        label="MetaMathQA",
        local_envs=("CARE_LORA_METAMATHQA_LOCAL", "METAMATHQA_LOCAL"),
    )


def _load_opencodeinstruct_train_split():
    """Load the OpenCodeInstruct train split with local/offline fallbacks."""
    use_streaming = str(os.environ.get("CARE_LORA_OPENCODEINSTRUCT_STREAMING", "true")).strip().lower()
    return _load_hub_split(
        "nvidia/OpenCodeInstruct",
        split="train",
        label="OpenCodeInstruct",
        local_envs=("CARE_LORA_OPENCODEINSTRUCT_LOCAL", "OPENCODEINSTRUCT_LOCAL"),
        streaming=use_streaming not in {"0", "false", "no", "off"},
    )


def _load_smoltalk_smol_magpie_ultra_train_split():
    """Load the SmolTalk smol-magpie-ultra train split with local/offline fallbacks."""
    use_streaming = str(os.environ.get("CARE_LORA_SMOLTALK_STREAMING", "false")).strip().lower()
    return _load_hub_split(
        "HuggingFaceTB/smoltalk",
        "smol-magpie-ultra",
        split="train",
        label="SmolTalk smol-magpie-ultra",
        local_envs=(
            "CARE_LORA_SMOLTALK_SMOL_MAGPIE_ULTRA_LOCAL",
            "CARE_LORA_SMOLTALK_LOCAL",
            "SMOLTALK_SMOL_MAGPIE_ULTRA_LOCAL",
            "SMOLTALK_LOCAL",
        ),
        streaming=use_streaming in {"1", "true", "yes", "on"},
    )


def _shuffle_for_filtering(dataset, *, seed: int, buffer_size: int = 10_000):
    """Shuffle Dataset/IterableDataset while keeping full-Dataset behavior unchanged."""
    try:
        return dataset.shuffle(seed=int(seed), buffer_size=int(buffer_size))
    except TypeError:
        return dataset.shuffle(seed=int(seed))


def _auto_tokenizer_from_pretrained(
    pretrained_model_name_or_path: str,
    *,
    use_fast: bool = True,
):
    """Load a tokenizer from the local override, the Hub, or the local HF cache."""
    from transformers import AutoTokenizer

    local = _env_path(
        "CARE_LORA_HF_MODEL_LOCAL",
        "CARE_LORA_TOKENIZER_LOCAL",
        "HF_MODEL_LOCAL",
        "TOKENIZER_LOCAL",
    )
    if local:
        if not os.path.isdir(local):
            raise FileNotFoundError(f"Tokenizer local path does not exist or is not a directory: {local!r}")
        log.info("Loading tokenizer from local path: %s", local)
        return AutoTokenizer.from_pretrained(
            local,
            use_fast=use_fast,
            local_files_only=True,
            trust_remote_code=True,
        )

    try:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=use_fast,
            trust_remote_code=True,
        )
    except OSError as error:
        if not _is_network_or_offline_error(error):
            raise
        log.warning(
            "Tokenizer download failed (%s); retrying with local Hugging Face cache.",
            error,
        )
        try:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=use_fast,
                local_files_only=True,
                trust_remote_code=True,
            )
        except OSError as cached_error:
            raise OSError(
                "Unable to load tokenizer from the Hub or the local Hugging Face cache. "
                "Set CARE_LORA_HF_MODEL_LOCAL or CARE_LORA_TOKENIZER_LOCAL to a directory "
                "containing tokenizer files.\n"
                f"Hub error: {error!r}\n"
                f"local_files_only retry: {cached_error!r}"
            ) from cached_error


def cache_to_disk(root_datadir):
    root_datadir = root_datadir if os.path.isabs(root_datadir) else os.path.join(_REPRODUCE_DIR, root_datadir)

    def _stable_cache_arg_key(func, args, kwargs) -> str:
        """Include call arguments in the cache key so dataset variants do not collide."""
        try:
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            payload = {k: bound.arguments[k] for k in sorted(bound.arguments.keys())}
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=repr)
        except Exception:
            encoded = repr((args, sorted(kwargs.items(), key=lambda kv: kv[0])))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]

    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            os.makedirs(root_datadir, exist_ok=True)
            func_name = func.__name__.replace("/", "")
            try:
                src = inspect.getsource(func)
                src_hash = hashlib.sha1(src.encode("utf-8")).hexdigest()[:10]
            except Exception:
                src_hash = "nosrc"
            args_hash = _stable_cache_arg_key(func, args, kwargs)
            cache_file = os.path.join(
                root_datadir,
                f"{func_name}__{DATA_CACHE_VERSION}__{src_hash}__{args_hash}.pkl",
            )

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    log.info("Loading cached data for %s", func.__name__)
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            log.info("Cached data for %s", func.__name__)
            return result

        return wrapper_cache

    return decorator_cache


@cache_to_disk("data_cache")
def load_sst2():
    dataset = _load_dataset_dict("glue", "sst2", label="GLUE/SST-2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_cola():
    dataset = _load_dataset_dict("glue", "cola", label="GLUE/CoLA")
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_mrpc():
    dataset = _load_dataset_dict("glue", "mrpc", label="GLUE/MRPC")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_mnli():
    dataset = _load_dataset_dict("glue", "mnli", label="GLUE/MNLI")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    return dataset["train"], dataset["validation_matched"], dataset["validation_matched"]


@cache_to_disk("data_cache")
def load_qnli():
    dataset = _load_dataset_dict("glue", "qnli", label="GLUE/QNLI")
    instruction = "classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    return dataset["train"], dataset["validation"], dataset["test"]


def _load_super_glue(config_name: str):
    return _load_dataset_dict("super_glue", config_name, label=f"SuperGLUE/{config_name}")


@cache_to_disk("data_cache")
def load_boolq():
    dataset = _load_super_glue("boolq")
    instruction = "answer the question about the passage with yes or no: "
    label_map = {0: "no", 1: "yes", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}passage: {e["passage"]}\nquestion: {e["question"]}\nresult: ',
            "y": label_map[int(e["label"])],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_cb():
    dataset = _load_super_glue("cb")
    instruction = "classify the relation between premise and hypothesis: "
    label_map = {0: "entailment", 1: "contradiction", 2: "neutral", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[int(e["label"])],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_copa():
    dataset = _load_super_glue("copa")

    def _row(e):
        relation = str(e.get("question", "cause"))
        prompt = (
            f"choose the better {relation} given the premise: {e['premise']}\n"
            f"alternative1: {e['choice1']}\n"
            f"alternative2: {e['choice2']}\n"
            "result: "
        )
        return {"x": prompt, "y": "first" if int(e["label"]) == 0 else "second"}

    dataset = dataset.map(_row)
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_rte():
    dataset = _load_super_glue("rte")
    instruction = "classify whether the hypothesis is entailed by the premise: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[int(e["label"])],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


@cache_to_disk("data_cache")
def load_wic():
    dataset = _load_super_glue("wic")
    instruction = "does the word have the same meaning in both sentences (answer yes or no): "
    label_map = {0: "no", 1: "yes", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": (
                f'{instruction}word: "{e["word"]}"\n'
                f'sentence1: {e["sentence1"]}\n'
                f'sentence2: {e["sentence2"]}\n'
                "result: "
            ),
            "y": label_map[int(e["label"])],
        }
    )
    return dataset["train"], dataset["validation"], dataset["validation"]


def _load_gsm8k_raw():
    return _load_dataset_dict("gsm8k", "main", label="GSM8K")


def load_gsm8k():
    """Load GSM8K main/train and main/test with a local processed cache."""
    marker = os.path.join(_GSM8K_CACHE, "dataset_dict.json")
    if os.path.isfile(marker):
        cached = load_from_disk(_GSM8K_CACHE)
        return cached["train"], cached["test"], cached["test"]

    raw = _load_gsm8k_raw()
    mapped = raw.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    os.makedirs(_PROCESSED_DATA_ROOT, exist_ok=True)
    to_save = DatasetDict({"train": mapped["train"], "test": mapped["test"]})
    to_save.save_to_disk(_GSM8K_CACHE)
    log.info("Saved processed GSM8K cache to %s", _GSM8K_CACHE)
    return mapped["train"], mapped["test"], mapped["test"]


_INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


@cache_to_disk(_METAMATHQA_CACHE)
def load_metamathqa(
    max_token_length: int = 512,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.3",
    shuffle_seed: int = 42,
    train_size: int = 100_000,
    val_size: int = 10_000,
):
    """MetaMathQA subset used for Mistral math training."""
    from tqdm import tqdm
    from utils import causal_lm_training_sequence_token_count

    raw = _load_metamath_qa_train_split().shuffle(seed=int(shuffle_seed))
    tokenizer = _auto_tokenizer_from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _row(sample):
        return {
            "x": f'Q: {sample["query"]}\nA: ',
            "y": str(sample.get("response", "") or "").strip(),
        }

    train_samples: tp.List[tp.Dict[str, str]] = []
    val_samples: tp.List[tp.Dict[str, str]] = []
    accepted = 0
    need = int(train_size) + int(val_size)

    for sample in tqdm(raw, desc="metamathqa(filter)"):
        if "GSM" not in str(sample.get("type", "")):
            continue
        row = _row(sample)
        ntok = causal_lm_training_sequence_token_count(row["x"], row["y"], tokenizer)
        if ntok >= int(max_token_length):
            continue
        if accepted < int(train_size):
            train_samples.append(row)
        elif accepted < need:
            val_samples.append(row)
        else:
            break
        accepted += 1

    if len(train_samples) < int(train_size) or len(val_samples) < int(val_size):
        log.warning(
            "metamathqa: expected train=%s val=%s, got train=%s val=%s",
            train_size,
            val_size,
            len(train_samples),
            len(val_samples),
        )

    val = Dataset.from_list(val_samples)
    return Dataset.from_list(train_samples), val, val


@cache_to_disk(_OPENCODEINSTRUCT_CACHE)
def load_opencodeinstruct(
    max_tokens: int = 1024,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.3",
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    train_size: int = 100_000,
    val_size: int = 10_000,
):
    """OpenCodeInstruct subset used for Mistral code training."""
    from tqdm import tqdm
    from utils import causal_lm_training_sequence_token_count

    raw = _shuffle_for_filtering(
        _load_opencodeinstruct_train_split(),
        seed=int(shuffle_seed),
        buffer_size=int(shuffle_buffer_size),
    )
    tokenizer = _auto_tokenizer_from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _row(sample):
        return {
            "x": _INSTRUCTION_TEMPLATE.format(instruction=str(sample.get("input", "")).strip()),
            "y": str(sample.get("output", "")).strip(),
        }

    train_samples: tp.List[tp.Dict[str, str]] = []
    val_samples: tp.List[tp.Dict[str, str]] = []
    accepted = 0
    need = int(train_size) + int(val_size)

    bar = tqdm(raw, desc="opencodeinstruct(filter)", total=None)
    for sample in bar:
        instruction = str(sample.get("input", "") or "").strip()
        response = str(sample.get("output", "") or "").strip()
        if not instruction or not response:
            continue
        row = _row(sample)
        ntok = causal_lm_training_sequence_token_count(row["x"], row["y"], tokenizer)
        if ntok >= int(max_tokens):
            continue
        bar.set_postfix(accepted=f"{accepted + 1}/{need}")
        if accepted < int(train_size):
            train_samples.append(row)
        elif accepted < need:
            val_samples.append(row)
        else:
            break
        accepted += 1

    if len(train_samples) < int(train_size) or len(val_samples) < int(val_size):
        log.warning(
            "opencodeinstruct: expected train=%s val=%s, got train=%s val=%s",
            train_size,
            val_size,
            len(train_samples),
            len(val_samples),
        )

    val = Dataset.from_list(val_samples)
    return Dataset.from_list(train_samples), val, val


def _normalize_chat_messages(messages) -> tp.Optional[tp.List[tp.Dict[str, str]]]:
    if not isinstance(messages, (list, tuple)):
        return None
    out: tp.List[tp.Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or "").strip().lower()
        content = str(msg.get("content", "") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out or None


def _last_assistant_index(messages: tp.Sequence[tp.Dict[str, str]]) -> tp.Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and str(messages[i].get("content", "")).strip():
            return i
    return None


def _manual_render_mistral_chat_template(
    messages: tp.Sequence[tp.Dict[str, str]],
    *,
    eos_token: str,
    add_generation_prompt: bool,
) -> str:
    """Fallback for base Mistral tokenizers that do not ship a chat_template."""
    eos = eos_token or "</s>"
    parts: tp.List[str] = []
    pending_system = ""
    first_user = True
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            pending_system = f"{pending_system}\n\n{content}".strip()
            continue
        if role == "user":
            user_content = f"{pending_system}\n\n{content}".strip() if pending_system else content
            pending_system = ""
            prefix = "<s>" if first_user else ""
            parts.append(f"{prefix}[INST] {user_content} [/INST]")
            first_user = False
            continue
        if role == "assistant":
            parts.append(f" {content}{eos}")
    if add_generation_prompt and parts and not parts[-1].endswith("[/INST]"):
        parts.append("")
    return "".join(parts)


_CHAT_TEMPLATE_FALLBACK_WARNED: set[str] = set()


def _chat_token_ids(
    tokenizer,
    messages: tp.Sequence[tp.Dict[str, str]],
    *,
    tokenizer_name: str,
    add_generation_prompt: bool,
) -> tp.List[int]:
    try:
        ids = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=bool(add_generation_prompt),
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]
    except Exception:
        if tokenizer_name not in _CHAT_TEMPLATE_FALLBACK_WARNED:
            _CHAT_TEMPLATE_FALLBACK_WARNED.add(tokenizer_name)
            log.warning(
                "tokenizer.apply_chat_template failed for %s; using the Mistral manual fallback.",
                tokenizer_name,
            )
        text = _manual_render_mistral_chat_template(
            messages,
            eos_token=getattr(tokenizer, "eos_token", None) or "</s>",
            add_generation_prompt=bool(add_generation_prompt),
        )
        return list(tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"])


def _common_prefix_len(a: tp.Sequence[int], b: tp.Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


def _assistant_label_spans(
    tokenizer,
    messages: tp.Sequence[tp.Dict[str, str]],
    full_ids: tp.Sequence[int],
    *,
    tokenizer_name: str,
) -> tp.List[tp.Tuple[int, int]]:
    spans: tp.List[tp.Tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if not str(msg.get("content", "")).strip():
            continue

        prefix_ids = _chat_token_ids(
            tokenizer,
            messages[:i],
            tokenizer_name=tokenizer_name,
            add_generation_prompt=True,
        )
        end_ids = _chat_token_ids(
            tokenizer,
            messages[: i + 1],
            tokenizer_name=tokenizer_name,
            add_generation_prompt=False,
        )
        start = len(prefix_ids)
        end = len(end_ids)
        if list(full_ids[:start]) != list(prefix_ids):
            start = _common_prefix_len(prefix_ids, full_ids)
        if list(full_ids[:end]) != list(end_ids):
            end = _common_prefix_len(end_ids, full_ids)
        if 0 <= start < end <= len(full_ids):
            spans.append((start, end))
    return spans


def _skip(stats: Counter, reason: str) -> None:
    stats[reason] += 1


def _build_pretokenized_sft_row(
    *,
    tokenizer,
    tokenizer_name: str,
    messages: tp.Sequence[tp.Dict[str, str]],
    assistant_label_scope: str,
) -> tp.Optional[tp.Dict[str, tp.List[int]]]:
    full_ids = _chat_token_ids(
        tokenizer,
        messages,
        tokenizer_name=tokenizer_name,
        add_generation_prompt=False,
    )
    spans = _assistant_label_spans(
        tokenizer,
        messages,
        full_ids,
        tokenizer_name=tokenizer_name,
    )
    if not spans:
        return None
    if assistant_label_scope == "last":
        spans = spans[-1:]

    labels = [-100] * len(full_ids)
    for start, end in spans:
        labels[start:end] = [int(x) for x in full_ids[start:end]]
    if all(int(x) == -100 for x in labels):
        return None
    return {
        "input_ids": [int(x) for x in full_ids],
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": int(len(full_ids)),
    }


@cache_to_disk(_SMOLTALK_SMOL_MAGPIE_ULTRA_CACHE)
def load_smoltalk_smol_magpie_ultra(
    max_tokens: int = 1024,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.3",
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    train_size: int = 100_000,
    val_size: int = 10_000,
    assistant_label_scope: str = "all",
):
    """SmolTalk smol-magpie-ultra subset used for Mistral instruction training."""
    from tqdm import tqdm

    assistant_label_scope = str(assistant_label_scope).strip().lower()
    if assistant_label_scope not in {"all", "last"}:
        raise ValueError("assistant_label_scope must be 'all' or 'last'.")

    raw = _shuffle_for_filtering(
        _load_smoltalk_smol_magpie_ultra_train_split(),
        seed=int(shuffle_seed),
        buffer_size=int(shuffle_buffer_size),
    )
    tokenizer = _auto_tokenizer_from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples: tp.List[tp.Dict[str, tp.List[int]]] = []
    val_samples: tp.List[tp.Dict[str, tp.List[int]]] = []
    accepted = 0
    need = int(train_size) + int(val_size)
    seen = 0
    stats: Counter = Counter()

    bar = tqdm(raw, desc="smoltalk_smol_magpie_ultra(filter)", total=None)
    for sample in bar:
        if accepted >= need:
            break
        seen += 1
        messages = _normalize_chat_messages(sample.get("messages"))
        if not messages:
            _skip(stats, "bad_or_empty_messages")
            continue

        first_user = next((m for m in messages if m["role"] == "user"), None)
        last_ai = _last_assistant_index(messages)
        if first_user is None or last_ai is None:
            _skip(stats, "missing_user_or_assistant")
            continue
        if not str(first_user.get("content", "")).strip():
            _skip(stats, "empty_first_user")
            continue

        full_messages = messages[: last_ai + 1]
        if not any(m["role"] == "user" for m in full_messages):
            _skip(stats, "missing_user_before_last_assistant")
            continue
        if not full_messages or full_messages[-1].get("role") != "assistant":
            _skip(stats, "last_retained_turn_not_assistant")
            continue
        if not str(full_messages[-1].get("content", "")).strip():
            _skip(stats, "empty_last_assistant")
            continue

        try:
            row = _build_pretokenized_sft_row(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                messages=full_messages,
                assistant_label_scope=assistant_label_scope,
            )
        except Exception:
            _skip(stats, "tokenize_or_span_exception")
            continue

        if row is None:
            _skip(stats, "no_assistant_label_span")
            continue
        if int(row["length"]) >= int(max_tokens):
            _skip(stats, "too_long")
            continue

        bar.set_postfix(accepted=f"{accepted + 1}/{need}")
        if accepted < int(train_size):
            train_samples.append(row)
        elif accepted < need:
            val_samples.append(row)
        accepted += 1

    if len(train_samples) < int(train_size) or len(val_samples) < int(val_size):
        log.warning(
            "smoltalk_smol_magpie_ultra: expected train=%s val=%s, got train=%s val=%s",
            train_size,
            val_size,
            len(train_samples),
            len(val_samples),
        )
    log.info(
        "smoltalk_smol_magpie_ultra: seen=%s accepted=%s train=%s val=%s max_tokens=%s tokenizer=%s "
        "assistant_label_scope=%s skipped=%s",
        seen,
        accepted,
        len(train_samples),
        len(val_samples),
        int(max_tokens),
        tokenizer_name,
        assistant_label_scope,
        dict(stats),
    )

    val = Dataset.from_list(val_samples)
    return Dataset.from_list(train_samples), val, val


DATASET_MAP = {
    "sst2": load_sst2,
    "cola": load_cola,
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "qnli": load_qnli,
    "boolq": load_boolq,
    "cb": load_cb,
    "copa": load_copa,
    "rte": load_rte,
    "wic": load_wic,
    "gsm8k": load_gsm8k,
    "metamathqa": load_metamathqa,
    "opencodeinstruct": load_opencodeinstruct,
    "smoltalk": load_smoltalk_smol_magpie_ultra,
    "smoltalk_smol_magpie_ultra": load_smoltalk_smol_magpie_ultra,
}


if __name__ == "__main__":
    print("Available datasets:")
    for name in sorted(DATASET_MAP):
        print(name)
