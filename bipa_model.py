"""ByT5 model operations for BIPA: training, inference, and evaluation."""

from __future__ import annotations

import inspect
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import regex
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger("bipa.model")
BASE_MODEL = "google/byt5-small"
BASE_MODEL_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
PUBLISHED_MODEL = "thiagomonteles/BIPA_g2p_Multidialect_Byt5"
PUBLISHED_MODEL_REVISION = "9068d5499496440873884c1e9ab91adaf971fdda"
IPA_GRAPHEMES = regex.compile(r"\X")


class G2PDataset:
    """Tokenized-on-demand examples in the model's dialect-tag input format."""

    def __init__(
        self, frame: pd.DataFrame, tokenizer: Any, max_input_length: int = 128
    ) -> None:
        self.inputs = (frame["dialect"] + " " + frame["word"]).tolist()
        self.targets = frame["pronunciation"].tolist()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        model_input = self.tokenizer(
            self.inputs[index],
            truncation=True,
            max_length=self.max_input_length,
        )
        target = self.tokenizer(
            text_target=self.targets[index],
            truncation=True,
            max_length=self.max_input_length,
        )
        model_input["labels"] = target["input_ids"]
        return model_input


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


def calculate_metrics(
    frame: pd.DataFrame,
    predicted_ids: Any,
    label_ids: Any,
    tokenizer: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate Minimum PER against all references for each dialect-word pair."""
    predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    labels = label_ids.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if len(predictions) != len(frame) or len(references) != len(frame):
        raise RuntimeError("Quantidade de predições/rótulos não corresponde ao split.")

    results: list[dict[str, Any]] = []
    dialect_scores: dict[str, list[tuple[float, int, int, bool]]] = defaultdict(list)
    groups = frame.groupby(["dialect", "word"], sort=False, dropna=False).indices

    for (dialect, word), raw_indices in groups.items():
        indices = list(raw_indices)
        predicted = predictions[indices[0]].strip()
        predicted_symbols = IPA_GRAPHEMES.findall(predicted)
        candidates = [references[index].strip() for index in indices]
        candidate_scores: list[tuple[float, int, int, str]] = []
        for reference in candidates:
            reference_symbols = IPA_GRAPHEMES.findall(reference)
            distance = levenshtein(predicted_symbols, reference_symbols)
            denominator = len(reference_symbols)
            rate = distance / denominator if denominator else float(distance > 0)
            candidate_scores.append((rate, distance, denominator, reference))

        best_rate, edits, ref_length, best_reference = min(
            candidate_scores, key=lambda score: score[0]
        )
        exact = predicted_symbols == IPA_GRAPHEMES.findall(best_reference)
        dialect_scores[dialect].append((best_rate, edits, ref_length, exact))
        results.append(
            {
                "dialect": dialect,
                "word": word,
                "prediction": predicted,
                "best_reference": best_reference,
                "num_references": len(candidates),
                "min_per": best_rate,
                "exact_match": exact,
            }
        )

    def summarize(
        scores: list[tuple[float, int, int, bool]],
    ) -> dict[str, float | int]:
        if not scores:
            return {
                "dialect_word_pairs": 0,
                "min_per": math.nan,
                "micro_per": math.nan,
                "exact_match_percent": math.nan,
            }
        ref_symbols = sum(score[2] for score in scores)
        return {
            "dialect_word_pairs": len(scores),
            "min_per": 100 * sum(score[0] for score in scores) / len(scores),
            "micro_per": (
                100 * sum(score[1] for score in scores) / ref_symbols
                if ref_symbols
                else math.nan
            ),
            "exact_match_percent": (
                100 * sum(score[3] for score in scores) / len(scores)
            ),
        }

    overall = summarize(
        [score for scores in dialect_scores.values() for score in scores]
    )
    metrics: dict[str, Any] = {
        **overall,
        "dialects": {
            dialect: summarize(scores)
            for dialect, scores in sorted(dialect_scores.items())
        },
    }
    return metrics, pd.DataFrame(results)


class BIPAG2PModel:
    """A loaded BIPA G2P model with training and inference operations."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    @classmethod
    def from_pretrained(
        cls, model_name: str, revision: str | None = None
    ) -> BIPAG2PModel:
        LOGGER.info("Carregando modelo: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        return cls(model, tokenizer, model_name)

    @classmethod
    def from_base_model(cls) -> BIPAG2PModel:
        return cls.from_pretrained(BASE_MODEL, BASE_MODEL_REVISION)

    def infer(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        num_beams: int = 6,
        max_length: int = 35,
    ) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size deve ser maior que zero.")
        if not texts:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        transcriptions: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                )
                transcriptions.extend(
                    self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                )
        return [text.strip() for text in transcriptions]

    def _trainer(
        self,
        eval_frame: pd.DataFrame,
        eval_batch_size: int,
        output_dir: Path,
        **training_options: Any,
    ) -> Seq2SeqTrainer:
        eval_dataset = G2PDataset(eval_frame, self.tokenizer)
        train_dataset = training_options.pop("train_dataset", None)
        arguments: dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_eval_batch_size": eval_batch_size,
            "predict_with_generate": True,
            "generation_num_beams": 6,
            "generation_max_length": 35,
            "report_to": [],
            "dataloader_pin_memory": True,
        }
        arguments.update(training_options)
        argument_names = inspect.signature(Seq2SeqTrainingArguments).parameters
        strategy_name = (
            "eval_strategy"
            if "eval_strategy" in argument_names
            else "evaluation_strategy"
        )
        if "evaluation_strategy" in arguments:
            arguments[strategy_name] = arguments.pop("evaluation_strategy")

        training_args = Seq2SeqTrainingArguments(**arguments)
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": training_args,
            "eval_dataset": eval_dataset,
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, model=self.model
            ),
            "compute_metrics": self._metric_callback(eval_frame),
        }
        if train_dataset is not None:
            trainer_kwargs["train_dataset"] = train_dataset
        trainer_parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
        tokenizer_parameter = "processing_class"
        if tokenizer_parameter not in trainer_parameters:
            tokenizer_parameter = "tokenizer"
        trainer_kwargs[tokenizer_parameter] = self.tokenizer
        return Seq2SeqTrainer(**trainer_kwargs)

    def _metric_callback(self, frame: pd.DataFrame):
        def compute_metrics(eval_prediction: Any) -> dict[str, float]:
            predictions = eval_prediction.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            metrics, _ = calculate_metrics(
                frame, predictions, eval_prediction.label_ids, self.tokenizer
            )
            return {
                "min_per": float(metrics["min_per"]),
                "micro_per": float(metrics["micro_per"]),
            }

        return compute_metrics

    def evaluate(
        self,
        frame: pd.DataFrame,
        output_dir: Path,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        trainer = self._trainer(frame, batch_size, output_dir)
        result = trainer.predict(trainer.eval_dataset)
        predicted_ids = (
            result.predictions[0]
            if isinstance(result.predictions, tuple)
            else result.predictions
        )
        metrics, predictions = calculate_metrics(
            frame, predicted_ids, result.label_ids, self.tokenizer
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)
        LOGGER.info(
            "%s: Minimum PER macro %.2f%%; micro %.2f%%",
            self.model_name,
            metrics["min_per"],
            metrics["micro_per"],
        )
        for dialect, dialect_metrics in metrics["dialects"].items():
            LOGGER.info(
                "%s: n=%s; Min PER=%.2f%%",
                dialect,
                dialect_metrics["dialect_word_pairs"],
                dialect_metrics["min_per"],
            )
        return metrics

    def train(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        output_dir: Path,
        seed: int,
        train_batch_size: int = 96,
        eval_batch_size: int = 16,
    ) -> None:
        set_seed(seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = G2PDataset(train_frame, self.tokenizer)
        if train_batch_size < 1:
            raise ValueError("train_batch_size deve ser maior que zero.")
        total_steps = (
            math.ceil(len(train_dataset) / train_batch_size) * 10
        )
        warmup_steps = int(total_steps * 0.1)

        training_options: dict[str, Any] = {
            "train_dataset": train_dataset,
            "num_train_epochs": 10,
            "learning_rate": 1e-4,
            "optim": "adafactor",
            "lr_scheduler_type": "linear",
            "warmup_steps": warmup_steps,
            "weight_decay": 0.05,
            "label_smoothing_factor": 0.1,
            "per_device_train_batch_size": train_batch_size,
            "per_device_eval_batch_size": eval_batch_size,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_min_per",
            "greater_is_better": False,
            "logging_strategy": "epoch",
            "seed": seed,
            "data_seed": seed,
        }
        trainer = self._trainer(
            validation_frame,
            eval_batch_size,
            output_dir,
            **training_options,
        )
        LOGGER.info(
            "Fine-tuning %s: treino=%s, validação=%s, teste=%s",
            BASE_MODEL,
            f"{len(train_frame):,}",
            f"{len(validation_frame):,}",
            f"{len(test_frame):,}",
        )
        trainer.train()
        self.model = trainer.model

        best_model_dir = output_dir / "best-model"
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(best_model_dir)
        (output_dir / "training_history.json").write_text(
            json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info(
            "Melhor checkpoint: %s (eval_min_per=%s)",
            trainer.state.best_model_checkpoint,
            trainer.state.best_metric,
        )
        for split_name, frame in (
            ("validation", validation_frame),
            ("test", test_frame),
        ):
            self.evaluate(
                frame,
                output_dir / split_name,
                batch_size=eval_batch_size,
            )
