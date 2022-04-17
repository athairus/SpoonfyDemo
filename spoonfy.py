#!/usr/bin/env python
# Spoonfy creates LitK ("literally-translated karaoke") subtitles from the given foreign-language video file in .ass format.
# Subtitles must be embedded in the input video.
# Usage: spoonfy.py <input-video>

import forcedalignment
import util
import extract
import literalkaraoke
import subtitles
from common import *


def main():
    in_fn, workdir = util.get_args()

    # Extract subtitles, audio to files in workdir
    extract.extract_subs(in_fn, workdir)
    extract.extract_audio(in_fn, workdir)

    # Load subtitles from workdir, expand start/end times earlier/later to ensure all words are present in audio
    spa, eng = extract.preprocess_subs(in_fn, workdir)

    # Split video into words, get start/end time for each
    words = forcedalignment.get_word_timings(in_fn, workdir, spa, eng)

    # Generate LitK for each word
    litk = literalkaraoke.generate_litk(workdir, words)

    # Generate output .ass subtitle file
    # FIXME: Get these params from argv
    desired_rate = 4.0
    min_speed = 0.5
    subtitles.generate_ass(in_fn, litk, desired_rate, min_speed)


if __name__ == "__main__":
    main()
