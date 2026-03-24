import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_LANGUAGE_PRESETS = {
    "facebook/nllb-200-distilled-600M": ("eng_Latn", "hun_Latn"),
    "facebook/m2m100_418M": ("en", "hu"),
    "Helsinki-NLP/opus-mt-en-hu": ("en", "hu"),
}


PROTECTED_PATTERN = re.compile(
    r"<[^>]+>"  # game markup tags like <color ...> or <ds-ficon ...>
    r"|\{\d+\}"  # numbered placeholders like {0}
    r"|%\d*\$?[sdif]"  # printf-style placeholders
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Death Stranding 2 localization JSON files with NLLB-200."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="localization.json.back",
        help="Path to the source localization JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the translated JSON file. Defaults to localization.hu.json next to the input file.",
    )
    parser.add_argument(
        "--source-lang",
        default="eng_Latn",
        help="NLLB source language code. Default: eng_Latn",
    )
    parser.add_argument(
        "--target-lang",
        default="hun_Latn",
        help="NLLB target language code. Default: hun_Latn",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="How many unique strings to translate per generation batch. Default: 12",
    )
    parser.add_argument(
        "--cache-file",
        help="Optional cache file path. If omitted, no cache file is loaded or written.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=60,
        help="Persist the cache after this many new translations. Default: 60",
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face model id to use. Default: facebook/nllb-200-distilled-600M",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster, lower-quality preset tuned for CPU-only runs.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generated token length per translated string. Default: 512",
    )
    return parser.parse_args()


def load_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent="\t")
        handle.write("\n")


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def needs_translation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    candidate = PROTECTED_PATTERN.sub("", stripped)
    return any(character.isalpha() for character in candidate)


def mask_protected_segments(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"ZXQPROTECT{counter:04d}END"
        replacements[token] = match.group(0)
        counter += 1
        return token

    return PROTECTED_PATTERN.sub(replace, text), replacements


def unmask_protected_segments(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, original in replacements.items():
        restored = restored.replace(token, original)
    return restored


def gather_unique_texts(payload: OrderedDict) -> list[str]:
    unique_texts: OrderedDict[str, None] = OrderedDict()

    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and needs_translation(text):
            unique_texts.setdefault(text, None)

    return list(unique_texts.keys())


class NllbTranslator:
    def __init__(self, model_name: str, source_lang: str, target_lang: str, max_length: int):
        tokenizer_kwargs = {}
        if model_name.startswith("facebook/"):
            tokenizer_kwargs["src_lang"] = source_lang

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model_name = model_name
        self.target_lang = target_lang
        self.max_length = max_length

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def translate_texts(self, texts: list[str]) -> list[str]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generate_kwargs = {
            "max_length": self.max_length,
        }
        if self.model_name.startswith("facebook/"):
            generate_kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(
                self.target_lang
            )

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                **generate_kwargs,
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def apply_fast_preset(args: argparse.Namespace) -> None:
    if not args.fast:
        return

    if args.model == "facebook/nllb-200-distilled-600M":
        args.model = "Helsinki-NLP/opus-mt-en-hu"
        if args.source_lang == "eng_Latn":
            args.source_lang = "en"
        if args.target_lang == "hun_Latn":
            args.target_lang = "hu"

    if args.batch_size == 12:
        args.batch_size = 48

    if args.max_length == 512:
        args.max_length = 192


def translate_batch(
    translator: NllbTranslator,
    texts: list[str],
) -> dict[str, str]:
    masked_texts: list[str] = []
    replacement_maps: list[dict[str, str]] = []

    for text in texts:
        masked, replacements = mask_protected_segments(text)
        masked_texts.append(masked)
        replacement_maps.append(replacements)

    results = translator.translate_texts(masked_texts)

    translated: dict[str, str] = {}
    for original, result, replacements in zip(texts, results, replacement_maps):
        translated[original] = unmask_protected_segments(result, replacements)

    return translated


def apply_translations(payload: OrderedDict, translations: dict[str, str]) -> None:
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text in translations:
            entry["text"] = translations[text]


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def default_output_path(input_path: Path) -> Path:
    if input_path.name.lower() == "localization.json.back":
        return input_path.with_name("localization.hu.json")

    return input_path.with_name(f"{input_path.stem}.hu{input_path.suffix}")


def main() -> int:
    args = parse_args()
    apply_fast_preset(args)

    if args.model in MODEL_LANGUAGE_PRESETS:
        default_source, default_target = MODEL_LANGUAGE_PRESETS[args.model]
        if args.model in {"facebook/m2m100_418M", "Helsinki-NLP/opus-mt-en-hu"}:
            if args.source_lang == "eng_Latn":
                args.source_lang = default_source
            if args.target_lang == "hun_Latn":
                args.target_lang = default_target

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )
    cache_path = Path(args.cache_file).expanduser().resolve() if args.cache_file else None

    payload = load_json(input_path)
    unique_texts = gather_unique_texts(payload)
    cache = load_cache(cache_path) if cache_path else {}

    print(f"Loading model: {args.model}", flush=True)
    translator = NllbTranslator(
        model_name=args.model,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_length=args.max_length,
    )

    pending = [text for text in unique_texts if text not in cache]
    total = len(unique_texts)
    print(f"Loaded {total} unique translatable strings.", flush=True)
    print(f"Cache hit: {total - len(pending)}", flush=True)
    print(f"Remaining: {len(pending)}", flush=True)

    new_translations = 0
    started_at = time.perf_counter()
    for batch_index, batch in enumerate(chunked(pending, args.batch_size), start=1):
        translated_batch = translate_batch(translator, batch)
        cache.update(translated_batch)
        new_translations += len(batch)

        translated_count = total - len(pending) + new_translations
        percentage = (translated_count / total * 100) if total else 100.0
        elapsed = time.perf_counter() - started_at
        if translated_count > 0 and elapsed > 0:
            remaining_count = total - translated_count
            seconds_per_string = elapsed / translated_count
            eta_seconds = remaining_count * seconds_per_string
            eta_text = f", ETA: {eta_seconds / 60:.1f}m"
        else:
            eta_text = ""
        print(
            f"Batch {batch_index}: translated {len(batch)} strings "
            f"({translated_count}/{total} total, {percentage:.2f}%). "
            f"Elapsed: {elapsed:.1f}s{eta_text}"
            ,
            flush=True,
        )

        if cache_path and new_translations % args.save_every == 0:
            save_json(cache_path, cache)
            print(f"Saved cache to {cache_path}", flush=True)

    apply_translations(payload, cache)
    save_json(output_path, payload)

    print(f"Saved translated file to {output_path}", flush=True)
    if cache_path:
        save_json(cache_path, cache)
        print(f"Saved cache to {cache_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())