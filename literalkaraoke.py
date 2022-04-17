import collections
import csv
import os
from pathlib import Path
import re
import sys
import typing
from typing import Callable, Union
import torch
import json

from transformers import BatchEncoding, pipeline
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration
from datasets import Dataset

from common import *
import forcedalignment

import ffmpeg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Returns a tokenizer function
def preprocess_with_tokenizer(
    tokenizer: M2M100Tokenizer,
    max_input_length: int,
    max_target_length: int,
) -> Callable[[dict[str, Union[list[str], list[list[str]]]], bool], BatchEncoding]:
    def preprocess(
        examples: dict[str, Union[list[str], list[list[str]]]], padding: bool = False
    ) -> BatchEncoding:
        # Spanish transcript, split by words
        spa_inputs: list[list[str]] = examples["Spanish"]
        # Reference translation (if available), a full sentence
        eng_inputs: list[str] = examples["Translation"]
        # LitK translation, split by corresponding Spanish words
        eng_targets: list[list[str]] = examples["English"]

        def groupify(example: list[str], reference: bool = False) -> str:
            # Position-encode inputs
            ret = []
            for i, word in enumerate(example):
                sep = f"[GRP{i}]"
                if reference:
                    sep = "[REF]"
                ret.append(f"{word}{sep} ")
            return "".join(ret)

        # Tokenize input pairs individually then join pairs together
        # NOTE: The model might need extra training to ignore the middle </s> token
        tokenizer.src_lang = "es"
        spa_inputs_tok = tokenizer.__call__(
            [groupify(s) for s in spa_inputs],
            # [" ".join(s) for s in spa_inputs],
            max_length=max_input_length // 2,
            truncation=True,
            padding=padding,
        )
        # print([tokenizer.convert_ids_to_tokens(ids) for ids in spa_inputs_tok["input_ids"]])
        tokenizer.src_lang = "en"
        eng_inputs_tok = tokenizer.__call__(
            [groupify(e.split(), reference=True) for e in eng_inputs],
            # eng_inputs,
            max_length=max_input_length // 2,
            truncation=True,
            padding=padding,
        )
        # print([tokenizer.convert_ids_to_tokens(ids) for ids in eng_inputs_tok["input_ids"]])
        model_inputs: BatchEncoding = {
            "input_ids": [
                # spa1 spa2 spa3 spa4 </s> eng1 eng2 </s>
                s + e
                for s, e in zip(
                    spa_inputs_tok["input_ids"], eng_inputs_tok["input_ids"]
                )
            ],
            "attention_mask": [
                s + e
                for s, e in zip(
                    spa_inputs_tok["attention_mask"], eng_inputs_tok["attention_mask"]
                )
            ],
        }

        # Tokenize targets
        targets = [groupify(eng) for eng in eng_targets]
        tokenizer.tgt_lang = "en"
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(  # [str] to [{'input_ids': [[int]], ...}]
                targets, max_length=max_target_length, truncation=True, padding=padding
            )
        # Must rename input_ids key to labels to differentiate input from target
        model_inputs["labels"] = labels["input_ids"]

        # For debugging
        # model_inputs["spa_in"] = [' '.join(s) for s in spa_inputs]
        # model_inputs["eng_in"] = [e for e in eng_inputs]
        # model_inputs["eng_out"] = [e for e in targets]

        return model_inputs  # {'input_ids': [[int]], 'labels': [[int]]}

    return preprocess


# Takes Spanish and English-reference lists of words
# Returns input to be passed directly to a model in inference mode (a dict with key "input_ids")
def preprocess_inference(
    preprocess: Callable[
        [dict[str, Union[list[str], list[list[str]]]], bool], BatchEncoding
    ],
    spanish: list[list[str]],
    references: list[str],
) -> dict[str, torch.Tensor]:
    ret = preprocess(
        {
            "Spanish": spanish,
            "Translation": references,
            "English": [""],
        },
        padding=True,
    )
    # Not needed for inference
    ret.pop("labels")
    ret.pop("attention_mask")

    # Upload to GPU
    ret["input_ids"] = torch.tensor(ret["input_ids"], dtype=torch.int, device=device)
    return ret


# Takes a batch of Spanish sentences (to get number of expected words) and model output, returns a batch of LitK sentences
def postprocess(transcripts: list[list[str]], outputs: list[str]) -> list[list[str]]:
    group_re = re.compile(r"(\[GRP\d+\])")
    group_num_re = re.compile(r"\[GRP(\d+)\]")
    output_slots = [[""] * len(transcript) for transcript in transcripts]
    cleaned_outputs = [
        output.replace("__en__", "").replace("</s>", "").replace("<pad>", "").strip()
        for output in outputs
    ]
    # print(cleaned_outputs)
    split_outputs = [
        # Ignore last one, it is (should be) an empty string
        [s.strip() for s in group_re.split(cleaned_output)][:-1]
        for cleaned_output in cleaned_outputs
    ]
    # print(f'{len(transcripts[0])=}')
    # print(split_outputs)
    for sent, slot in zip(split_outputs, output_slots):
        prev_output = None
        for out in sent:
            group_num_match = group_num_re.match(out)
            # If end of a word group
            if group_num_match:
                group_num = int(group_num_match.group(1))
                # If the model didn't mess up & give us some OoB value for this input, then the last token we saw was the word group
                if group_num < len(slot):
                    slot[group_num] = prev_output
            prev_output = out

    return output_slots


# Takes a batch of Spanish sentences (split by words) and a batch of English reference sentences and returns a batch
# of English LitK translations (split by corresponding Spanish word)
def translate_litk(
    model: M2M100ForConditionalGeneration,
    tokenizer: M2M100Tokenizer,
    preprocess: Callable[
        [dict[str, Union[list[str], list[list[str]]]], bool], BatchEncoding
    ],
    spanish: list[list[str]],
    references: list[str],
) -> list[list[str]]:
    input_ids = preprocess_inference(preprocess, spanish, references)
    generated_tokens: torch.Tensor = model.generate(
        **input_ids, forced_bos_token_id=tokenizer.get_lang_id("en")
    )
    return postprocess(
        spanish, tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    )

# Entry point. Takes a list of word groups and adds LitK translations to each group 
def generate_litk(workdir: Path, groups: list[WordGroup]) -> list[WordGroup]:
    # Init model
    max_input_length = 256  # Token IDs (ints)
    max_target_length = 256  # Token IDs (ints)
    spoonfy_model_name = "athairus/m2m100_418M_finetuned_litk_es_en"
    print("Loading translation model...")
    translator = pipeline("translation", model=spoonfy_model_name)
    model, tokenizer = translator.model, translator.tokenizer
    model.train(False)
    model.to(device)  # Upload model to GPU
    preprocess = preprocess_with_tokenizer(
        tokenizer, max_input_length, max_target_length
    )
    print("Model loaded.")
    # Build sentences out of word groups
    sents: list[list[WordGroup]] = []
    curr_sent = []
    for group in groups:
        curr_sent.append(group)
        if group.end_of_sentence:
            sents.append(curr_sent)
            curr_sent = []
    sents.append(curr_sent)
    # Sort sentences by source length so we waste as little computation on padding tokens as possible per batch
    # FIXME: Sort by length of tokenized subwords, not words
    sents.sort(key=lambda sent: len(sent))
    sents.reverse()

    # Translate batches
    batch_size = 24
    for i in tqdm(range(0, len(sents), batch_size)):
        batch: list[list[WordGroup]] = sents[i : i + batch_size]
        spa_batch = [[group.source for group in sent] for sent in batch]
        # Assemble relevant English subtitles from the video
        # Gather nonempty references
        eng_refs_batch = [
            [group.reference.words for group in sent if group.reference]
            for sent in batch
        ]
        # Dedupe (can't use set, doesn't preserve order)
        eng_refs_batch = [dict.fromkeys(eng_refs) for eng_refs in eng_refs_batch]
        # Assemble into a single string per batch item
        eng_refs_batch = [" ".join(eng_refs.keys()) for eng_refs in eng_refs_batch]

        # ref_batch = ["" for sent in batch]
        litk_batch = translate_litk(
            model, tokenizer, preprocess, spa_batch, eng_refs_batch
        )
        # print(spa_batch)
        # print(eng_refs_batch)
        # print(litk_batch)
        for sent, litk_sent in zip(batch, litk_batch):
            for group, litk in zip(sent, litk_sent):
                group.target = litk.strip()

                # Small hack to work around wonky single-word translations
                # If the reference is just one word, there's no ambiguity. We can use the reference
                # translation as LitK directly!
                if (
                    len(group.reference.words.strip().split()) == 1
                    and group.end_of_sentence
                ):
                    group.target = group.reference.words.strip()
                    # print(f'Chose reference translation {group.target} over machine translation {litk.strip()} for word {group.source}')

    # FIXME: For debugging, delete once done w/ everything. 
    # Two output formats: .tsv and .json. Uncomment one or both blocks below 
    # # Write .tsv containing 1 Spanish word per row
    # print("Writing .tsv file...")
    # tsv_fn = workdir / f"litk.tsv"
    # with open(tsv_fn, "w") as tsv:
    #     # Header
    #     fieldnames = ["Start", "End", "Spanish", "LitK", "English"]
    #     writer = csv.DictWriter(tsv, fieldnames=fieldnames, dialect="excel-tab")
    #     writer.writeheader()
    #     for group in groups:
    #         if not group.target:
    #             # print(f'Skipping blank target, {group=}')
    #             continue
    #         eng_hint = "Y" if group.end_of_sentence else ""
    #         writer.writerow(
    #             {
    #                 "Start": f"{group.start}",
    #                 "End": f"{group.end}",
    #                 "Spanish": group.source,
    #                 "LitK": group.target,
    #                 "English": eng_hint,
    #             }
    #         )

    # json_fn = workdir / f"litk.json"
    # json_groups = []
    # for i, group in enumerate(groups):
    #     json_groups.append(
    #         {
    #             "start": group.start,
    #             "end": group.end,
    #             "spanish": group.source,
    #             "litk": group.target,
    #             "english": "",  # TODO
    #             "id": i,
    #         }
    #     )
    # with open(json_fn, "w") as j:
    #     json.dump(json_groups, j, indent=4)

    return groups
