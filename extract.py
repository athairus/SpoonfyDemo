import os
from pathlib import Path
import subprocess
import sys

import ffmpeg
import srt

from common import *
import util


def extract_subs(video: Path, dest: Path):
    print("Extracting subs...")
    video = str(video)

    for lang in ["eng", "spa"]:
        fn = dest / f"subtitles-{lang}"
        if os.path.exists(f"{fn}.srt"):
            print(f'"{fn}.srt" exists, skipping subtitle extraction')
            continue

        print(f"Extracting {lang} subtitles...")

        # Are these subs DVD or Blu-ray bitmapped subs? If so, they must go through an OCR
        # Figure that out now
        is_bitmapped = False
        codec_name = ""
        index = 0  # Used to select a single subtitle
        probe = ffmpeg.probe(video)
        # print(probe)

        for stream in probe["streams"]:
            if stream["codec_type"] != "subtitle":
                continue
            if stream["tags"]["language"] != lang:
                continue
            index = stream["index"]
            codec_name = stream["codec_name"]
            if (
                codec_name == "hdmv_pgs_subtitle" or codec_name == "vob"
            ):  # TODO: Find out vobsub codec name
                is_bitmapped = True
                break

        if is_bitmapped:
            print("FIXME: Need a good OCR solution for Blu-ray/DVD subs. Exiting...")
            sys.exit(1)

        else:
            (
                ffmpeg.input(video)[
                    f"{index}"
                ]  # Open input, select first Spanish/English sub stream
                .output(
                    f"{fn}.srt"
                )  # Output .srt file w/ same base name in same folder
                .overwrite_output()
                .run(quiet=True)
            )


def extract_audio(video: Path, dest: Path):
    print("Extracting audio...")
    video, dest = str(video), str(dest / "audio")

    if os.path.exists(f"{dest}.wav"):
        print(f'"{dest}.wav" exists, skipping audio extraction')
        return

    (
        ffmpeg.input(video)
        # .filter('pan', '5.1| FC < 4 * FC')  # This effectively multiples the other channels by 1/4 due to normalization (< vs =)
        # .filter(
        #     'arnndn',
        #     model='/path/to/rnnoise-models/leavened-quisling-2018-08-31/lq.rnnn',
        #     mix=0.5
        # )
        .output(f"{dest}.wav", ac=1, ar="16k").run(quiet=True)
    )

    return


def preprocess_subs(
    video: Path, workdir: Path
) -> tuple[list[Subtitle], list[Subtitle]]:
    # Load subtitles into memory
    fn = str(workdir / "subtitles")
    spa_raw = Path(f"{fn}-spa.srt").read_text()
    eng_raw = Path(f"{fn}-eng.srt").read_text()
    spa_objs = list(srt.parse(spa_raw))
    eng_objs = list(srt.parse(eng_raw))
    spa_subs: list[Subtitle] = []
    eng_subs: list[Subtitle] = []

    def scrub_sub(sub):
        # No italics!
        sub = sub.replace("<i>", "")
        sub = sub.replace("</i>", "")
        sub = sub.replace("♪", "")
        # Ignore newlines
        sub = sub.replace("\n", " ")
        sub = sub.strip()
        return sub

    for obj in spa_objs:
        obj: srt.Subtitle
        sub = scrub_sub(obj.content)
        start = obj.start.total_seconds()
        end = obj.end.total_seconds()
        # Ignore subtitles that aren't dialogue (e.g. for the hard-of-hearing)
        # TODO: Replace with a model. Why not?
        scrubbed_sub = []
        if sub and (
            (
                sub.startswith("(")
                or sub[1:].startswith("(")
                or sub.startswith("[")
                or sub[1:].startswith("[")
            )
            or (sub.endswith(")") or sub.endswith("]"))
        ):
            continue
        for word in sub.split():
            if not word:
                continue
            if word and (
                (
                    word.startswith("(")
                    or word[1:].startswith("(")
                    or word.startswith("[")
                    or word[1:].startswith("[")
                )
                or word.endswith(")" or word.endswith("]"))
            ):
                continue
            scrubbed_sub.append(word)
        scrubbed_sub = " ".join(scrubbed_sub)
        if scrubbed_sub:
            spa_subs.append(Subtitle(scrubbed_sub, start, end))

    for obj in eng_objs:
        obj: srt.Subtitle
        sub = scrub_sub(obj.content)
        start = obj.start.total_seconds()
        end = obj.end.total_seconds()
        eng_subs.append(Subtitle(sub, start, end))

    # Expand Spanish start, end times by a few seconds and add dummy start/end words
    # to ensure the forced aligner is able to see the entire first/last word (which are now not at the edge of the sub)
    expand_by = 2
    for sub in spa_subs[1:]:
        sub.start = sub.start - expand_by
    for sub in spa_subs[:-1]:
        sub.end = sub.end + expand_by
    # # Make a copy to use as a reference as we're modifying the original
    # spa_subs_ref = [Subtitle(sub.words, sub.start, sub.end) for sub in spa_subs]
    # for i, sub in enumerate(spa_subs):
    #     # First sub: Extend end time to a bit into next sub (so that the last word is included in the audio snippet)
    #     if sub == spa_subs[0]:
    #         next_sub = spa_subs_ref[i + 1]
    #         next_word = next_sub.words.split()[0]
    #         sub.words = f"START {sub.words} {next_word}"
    #         sub.end = next_sub.start + expand_by
    #     # Last sub: Retract start time to a bit into prev sub (same idea but first word)
    #     elif sub == spa_subs[-1]:
    #         prev_sub = spa_subs_ref[i - 1]
    #         prev_word = prev_sub.words.split()[-1]
    #         sub.words = f"{prev_word} {sub.words} END"
    #         sub.start = prev_sub.end - expand_by
    #     # All other subs: Do both
    #     else:
    #         prev_sub = spa_subs_ref[i - 1]
    #         next_sub = spa_subs_ref[i + 1]
    #         prev_word = prev_sub.words.split()[-1]
    #         next_word = next_sub.words.split()[0]
    #         sub.words = f"{prev_word} {sub.words} {next_word}"
    #         sub.start = prev_sub.end - expand_by
    #         sub.end = next_sub.start + expand_by

    # Handle overlaps
    for i in range(len(spa_subs) - 1):
        left, right = spa_subs[i].end, spa_subs[i + 1].start
        if left > right:
            mid = (left - right) / 2
            spa_subs[i].end = spa_subs[i].end - mid  # Move end backward
            spa_subs[i + 1].start = spa_subs[i + 1].start + mid  # Move start forward

    # for sub in spa_subs[:10]:
    #     print(sub)

    return spa_subs, eng_subs
