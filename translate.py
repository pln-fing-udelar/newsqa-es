#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="cnn_stories/cnn/stories")
    parser.add_argument("--output-path", default="cnn_stories/cnn/translated")

    parser.add_argument("--model-name", default="Helsinki-NLP/opus-mt-en-es")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--batch-size", type=int, default=46)
    parser.add_argument("--num-workers", type=int,
                        default=len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1))
    return parser.parse_args()


def collate(instances: Sequence[Mapping[str, Any]],
            tokenizer: PreTrainedTokenizerBase | None = None) -> Mapping[str, Any]:
    keys = next(iter(instances), {})
    batch = {k: [instance[k] for instance in instances] for k in keys}

    if tokenizer:
        for k in keys:
            stack = batch[k]

            if k == "line":
                tokenized = tokenizer(stack, truncation=True, padding=True, return_tensors="pt")
                batch[f"{k}_ids"] = tokenized["input_ids"]
                batch[f"{k}_mask"] = tokenized["attention_mask"]

    return batch


# We translate the files line by line because the files may be too large, and the lines (sentences) are the best
# breaking points. To optimize the use of the device memory, we batch by line. However, we need to detect when a file
# ends to save it.
#
# Another consequence is that it's better to use an iterable-style dataset because with a map-style dataset we would
# need to know beforehand the (non-empty) line count from each file to then process them. This can be saved by using
# a custom sampler that checks the file size as it's being iterated., Still, with a map-style dataset we would re-open
# the same files over and over in the `__getitem__` method, for each line. With an iterable-style dataset we don't
# have these issues.
#
# But there are other issues with the iterable-style dataset. To not be bottlenecked by the I/O, we use multiple
# workers. Note that in iterable-style datasets the workers need to be handled manually. We need to choose what each
# worker returns. Moreover, since we care about the order of the results (because we want to process all lines in
# order and save the corresponding file), we need to know the batch size and count how many items are being return by
# each worker's batch.
class TextFileLineDataset(IterableDataset):
    def __init__(self, paths: Iterable[str], batch_size: int) -> None:
        self.paths = paths
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        i = 0

        worker_info = torch.utils.data.get_worker_info()
        paths, self.paths = itertools.tee(self.paths)
        for path in paths:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    if line := line.strip():
                        # Using an iterable dataset with multiple workers requires to manually select which
                        # instances are returned.
                        if worker_info is None or worker_info.id == (i // self.batch_size) % worker_info.num_workers:
                            yield {"line": line, "path": path}
                        i += 1


def _write_file_maybe(folder: str, filename: str | None, lines: Iterable[str]) -> None:
    if filename is not None:
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.eval()

    paths = (os.path.join(args.data_path, filename)
             for filename in os.listdir(args.data_path)
             if not os.path.exists(os.path.join(args.output_path, filename)))  # We only compute the missing files.
    dataset = TextFileLineDataset(paths=paths, batch_size=args.batch_size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                             collate_fn=lambda instances: collate(instances, tokenizer=tokenizer))

    file_count = len(next(os.walk(args.data_path))[2])  # More efficient than `listdir` as it doesn't sort.
    done_file_count = len(next(os.walk(args.output_path))[2])

    with torch.inference_mode(), \
            tqdm(initial=done_file_count, total=file_count, desc="Translating", unit="file") as progress:
        last_filename = None
        translated_lines = None

        for batch in data_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            generated_token_ids = model.generate(batch["line_ids"], attention_mask=batch["line_mask"])
            translated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

            for translated_line, path in zip(translated_text, batch["path"]):
                filename = os.path.basename(path)

                if last_filename != filename:  # A new file starts.
                    _write_file_maybe(args.output_path, last_filename, translated_lines)

                    last_filename = filename
                    translated_lines = []

                    progress.update()

                translated_lines.append(translated_line)

        _write_file_maybe(args.output_path, last_filename, translated_lines)

        progress.update()


if __name__ == "__main__":
    main()
