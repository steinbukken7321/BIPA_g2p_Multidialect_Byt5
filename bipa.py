"""Run inference and evaluate the published BIPA grapheme-to-phoneme model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import regex
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "thiagomonteles/BIPA_g2p_Multidialect_Byt5"
MODEL_REVISION = "9068d5499496440873884c1e9ab91adaf971fdda"
DATASET_ID = "thiagomonteles/BIPA"
DATASET_FILE = "por_bz_com_dialeto_55K.tsv"
DATASET_REVISION = "d1e81df1565756229bc20d6841716a7a5560fc80"
SPLIT_SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent
IPA_GRAPHEMES = regex.compile(r"\X")
DIALECTS = (
    "[brasil]",
    "[rio-de-janeiro]",
    "[são-paulo]",
    "[regiao-sul]",
    "[regiao-nordeste]",
    "[regiao-centro-oeste]",
)
DIALECT_ALIASES = {
    "brazil": "[brasil]",
    "brazil general": "[brasil]",
    "rio de janeiro": "[rio-de-janeiro]",
    "sao paulo": "[são-paulo]",
    "southern brazil": "[regiao-sul]",
    "south region": "[regiao-sul]",
    "south": "[regiao-sul]",
    "northeast brazil": "[regiao-nordeste]",
    "northeast region": "[regiao-nordeste]",
    "central west brazil": "[regiao-centro-oeste]",
    "center west brazil": "[regiao-centro-oeste]",
    "center west region": "[regiao-centro-oeste]",
    "central west region": "[regiao-centro-oeste]",
    "rural central brazil": "[regiao-centro-oeste]",
}


def load_model() -> tuple[Any, Any, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Carregando %s (%s) em %s", MODEL_ID, MODEL_REVISION, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(device)
    model.eval()
    return model, tokenizer, device


def generate(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    inputs: Sequence[str],
    batch_size: int,
) -> list[str]:
    predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            encoded = tokenizer(
                list(inputs[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            output_ids = model.generate(
                **encoded,
                num_beams=6,
                max_length=35,
            )
            predictions.extend(
                text.strip()
                for text in tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
            )
    return predictions


def normalize_dialect(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = " ".join(
        normalized.casefold()
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", "")
        .replace(")", "")
        .split()
    )
    try:
        return DIALECT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Rótulo de dialeto não reconhecido no dataset: {label!r}") from exc


def download_dataset() -> pd.DataFrame:
    dataset_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    frame = pd.read_csv(dataset_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"dialeto", "palavra", "pronuncia"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no TSV do BIPA: {', '.join(sorted(missing))}"
        )
    frame = frame.rename(
        columns={
            "dialeto": "dialect_label",
            "palavra": "word",
            "pronuncia": "pronunciation",
        }
    )
    frame["dialect"] = frame["dialect_label"].map(normalize_dialect)
    frame = frame.loc[
        frame["word"].str.strip().ne("")
        & frame["pronunciation"].str.strip().ne("")
    ].copy()
    frame = frame.drop_duplicates(["dialect", "word", "pronunciation"])
    return frame[["dialect", "word", "pronunciation"]].reset_index(drop=True)


def create_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create group-safe 75/15/10 splits stratified by dialect and word length."""
    strata = frame["dialect"] + "|" + frame["word"].str.len().astype(str)
    groups = pd.factorize(
        pd.MultiIndex.from_frame(frame[["dialect", "word"]])
    )[0]
    test_splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=SPLIT_SEED
    )
    remaining_indices, test_indices = next(
        test_splitter.split(frame[["word"]], strata, groups)
    )
    remaining = frame.iloc[remaining_indices].reset_index(drop=True)
    remaining_strata = (
        remaining["dialect"] + "|" + remaining["word"].str.len().astype(str)
    )
    remaining_groups = pd.factorize(
        pd.MultiIndex.from_frame(remaining[["dialect", "word"]])
    )[0]
    validation_splitter = StratifiedGroupKFold(
        n_splits=6, shuffle=True, random_state=SPLIT_SEED
    )
    train_indices, validation_indices = next(
        validation_splitter.split(
            remaining[["word"]], remaining_strata, remaining_groups
        )
    )
    splits = {
        "train": remaining.iloc[train_indices].reset_index(drop=True),
        "validation": remaining.iloc[validation_indices].reset_index(drop=True),
        "test": frame.iloc[test_indices].reset_index(drop=True),
    }
    pair_sets = {
        name: set(zip(part["dialect"], part["word"], strict=False))
        for name, part in splits.items()
    }
    split_names = tuple(splits)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if pair_sets[first] & pair_sets[second]:
                raise RuntimeError(f"Vazamento entre {first} e {second}.")
    return splits


def levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def score_predictions(
    pairs: pd.DataFrame,
    predictions: Sequence[str],
    references: dict[tuple[str, str], list[str]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(pairs) != len(predictions):
        raise RuntimeError("Quantidade de previsões não corresponde às palavras.")

    scored: list[dict[str, Any]] = []
    by_dialect: dict[str, list[tuple[float, int, int, bool]]] = defaultdict(list)
    for pair, prediction in zip(
        pairs.itertuples(index=False), predictions, strict=True
    ):
        candidate_scores: list[tuple[float, int, int, str]] = []
        predicted_symbols = IPA_GRAPHEMES.findall(prediction.strip())
        candidates = references[(pair.dialect, pair.word)]
        for reference in candidates:
            reference_symbols = IPA_GRAPHEMES.findall(reference.strip())
            edits = levenshtein(predicted_symbols, reference_symbols)
            length = len(reference_symbols)
            rate = edits / length if length else float(edits > 0)
            candidate_scores.append((rate, edits, length, reference.strip()))
        best_rate, edits, reference_length, best_reference = min(
            candidate_scores, key=lambda item: (item[0], item[1], item[2])
        )
        exact = predicted_symbols == IPA_GRAPHEMES.findall(best_reference)
        by_dialect[pair.dialect].append(
            (best_rate, edits, reference_length, exact)
        )
        scored.append(
            {
                "dialect": pair.dialect,
                "word": pair.word,
                "prediction": prediction.strip(),
                "best_reference": best_reference,
                "num_references": len(candidates),
                "min_per_percent": 100 * best_rate,
                "exact_match": exact,
            }
        )

    def summarize(
        scores: list[tuple[float, int, int, bool]],
    ) -> dict[str, float | int]:
        if not scores:
            return {
                "dialect_word_pairs": 0,
                "min_per_percent": float("nan"),
                "micro_per_percent": float("nan"),
                "exact_match_percent": float("nan"),
            }
        total_reference_symbols = sum(score[2] for score in scores)
        return {
            "dialect_word_pairs": len(scores),
            "min_per_percent": 100 * sum(score[0] for score in scores) / len(scores),
            "micro_per_percent": (
                100 * sum(score[1] for score in scores) / total_reference_symbols
                if total_reference_symbols
                else float("nan")
            ),
            "exact_match_percent": (
                100 * sum(score[3] for score in scores) / len(scores)
            ),
        }

    overall = summarize(
        [score for dialect_scores in by_dialect.values() for score in dialect_scores]
    )
    metrics = {
        **overall,
        "dialects": {
            dialect: summarize(scores)
            for dialect, scores in sorted(by_dialect.items())
        },
    }
    return metrics, pd.DataFrame(scored)


def run_evaluation(args: argparse.Namespace) -> None:
    frame = download_dataset()
    splits = create_splits(frame)
    test_frame = splits[args.split]
    pairs = test_frame[["dialect", "word"]].drop_duplicates().reset_index(drop=True)
    if args.dialect:
        pairs = pairs.loc[pairs["dialect"].eq(args.dialect)].reset_index(drop=True)
    if pairs.empty:
        raise ValueError(
            f"Não há pares para o split {args.split} e o dialeto informado."
        )
    is_subset = args.limit is not None and args.limit < len(pairs)
    if is_subset:
        pairs = pairs.sample(n=args.limit, random_state=SPLIT_SEED).reset_index(drop=True)

    references = {
        (dialect, word): group["pronunciation"].tolist()
        for (dialect, word), group in test_frame.groupby(
            ["dialect", "word"], sort=False
        )
    }
    model, tokenizer, device = load_model()
    model_inputs = [f"{dialect} {word}" for dialect, word in pairs.itertuples(index=False)]
    predictions = generate(
        model, tokenizer, device, model_inputs, args.batch_size
    )
    metrics, scored = score_predictions(pairs, predictions, references)
    metrics.update(
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": args.split,
            "split_seed": SPLIT_SEED,
            "split_examples": {
                name: len(part) for name, part in splits.items()
            },
            "split_unique_pairs": {
                name: int(part[["dialect", "word"]].drop_duplicates().shape[0])
                for name, part in splits.items()
            },
            "evaluated_pairs": len(pairs),
            "is_subset": is_subset,
            "dialect_filter": args.dialect,
            "metric_note": (
                "PER mínimo por par dialeto-palavra, com símbolos IPA "
                "segmentados em grafemas Unicode estendidos."
            ),
        }
    )
    output_name = args.split
    if args.dialect:
        output_name += "-" + args.dialect.strip("[]")
    if is_subset:
        output_name += f"-sample-{len(pairs)}"
    output_dir = PROJECT_DIR / args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    scored.to_csv(
        output_dir / "predictions.tsv", sep="\t", index=False, encoding="utf-8"
    )
    logging.info(
        "Minimum PER macro: %.2f%% (%s pares); resultados em %s",
        metrics["min_per_percent"],
        metrics["evaluated_pairs"],
        output_dir.resolve(),
    )
    for dialect, result in metrics["dialects"].items():
        logging.info(
            "%s: Minimum PER %.2f%% (%s pares)",
            dialect,
            result["min_per_percent"],
            result["dialect_word_pairs"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Usa ou avalia o modelo BIPA já publicado no Hugging Face."
    )
    parser.add_argument("--evaluate", action="store_true", help="Avalia no dataset BIPA.")
    parser.add_argument("--dialect", choices=DIALECTS)
    parser.add_argument("--text", nargs="+", metavar="PALAVRA")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        help="Avalia uma amostra aleatória para testar o fluxo; não é resultado completo.",
    )
    parser.add_argument("--output-dir", default="runs/evaluation")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size deve ser maior que zero.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit deve ser maior que zero.")
    if args.evaluate:
        if args.text:
            parser.error("--text é usado para inferência, não junto com --evaluate.")
    elif not args.dialect or not args.text:
        parser.error("Para inferência, informe --dialect e --text.")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.evaluate:
        run_evaluation(args)
        return

    model, tokenizer, device = load_model()
    inputs = [f"{args.dialect} {word}" for word in args.text]
    predictions = generate(model, tokenizer, device, inputs, args.batch_size)
    for word, prediction in zip(args.text, predictions, strict=True):
        print(f"{args.dialect} {word}\t{prediction}")


if __name__ == "__main__":
    main()
