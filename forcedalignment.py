from dataclasses import dataclass
from pathlib import Path
import shutil

from transformers import Wav2Vec2Model

from common import *
import asrtest

import ffmpeg
from pydub import AudioSegment


def get_word_timings(
    video: Path,
    workdir: Path,
    spa_subs: list[Subtitle],
    eng_subs: list[Subtitle],
) -> list[WordGroup]:
    # Get video framerate
    fps = 0
    probe = ffmpeg.probe(str(video))
    for stream in probe["streams"]:
        codec_type = stream["codec_type"]
        if codec_type == "video":
            fps_str = stream["avg_frame_rate"]
            fps_frac = fps_str.split("/")
            fps = float(fps_frac[0])
            if len(fps_frac) > 1:
                fps = fps / float(fps_frac[1])

    # Get timing for every word in every subtitle
    groups: list[WordGroup] = []
    print("Getting start/end times for every word in video")
    audio = AudioSegment.from_wav(f'{str(workdir / "audio")}.wav')
    processor, model = asrtest.load()
    for sub in tqdm(spa_subs):
        sub: Subtitle

        # Extract audio snippet
        start, end = sub.start, sub.end
        wav_dir = workdir / "audio-segments"
        wav_dir.mkdir(exist_ok=True)
        wav_fn = f"{str(wav_dir / str(start))}.wav"
        start, end = int(start * 10**3), int(end * 10**3)
        if not Path(wav_fn).exists():
            audio[start:end].export(wav_fn, format="wav")
        sent_offset = start / 10**3
        group_offset = len(groups)

        # Preprocess subtitle
        preproc_sent = []
        words = sub.words.split()
        for i, word in enumerate(words):
            word: str
            group = WordGroup(word)
            # # Mark dummy words as fillers
            # if i == 0 or i == len(words) - 1:  
            #     group.filler = True
            groups.append(group)
            word = "".join([c for c in word if c.isalpha()])
            
            # To keep the length of the sentence (when split by spaces) constant, replace empty strings with a dummy char
            if not word:
                word = "-"
            preproc_sent.append(word)
        preproc_sent = " ".join(preproc_sent)

        # Get start, end times
        # TODO: Batch inputs
        times = asrtest.get_timings(processor, model, [wav_fn], workdir, [preproc_sent], [sub.start])[0]
        for i in range(len(times)):
            start, end = times[i]
            groups[i + group_offset].start = start + sent_offset
            groups[i + group_offset].end = end + sent_offset

            # Find most relevant English subtitle, keep a reference to it
            mid = (start + end) / 2
            mid = mid + sent_offset
            # This is O(n^2), not a problem at the scale we're working at here (<1000 subtitles per video file)
            for eng_sub in eng_subs:
                if eng_sub.start < mid:
                    groups[i + group_offset].reference = eng_sub

        # # Get rid of the dummy words (they served their purpose)
        # groups = [group for group in groups if not group.filler]

    # asrtest.unload()

    # Generate sentence breaks
    # TODO: Use NLTK to do this more intelligently
    for group in groups:
        group.end_of_sentence = sent_enders.match(group.source) is not None

    return groups
