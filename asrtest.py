from common import *

from pathlib import Path
import torch
import librosa
from datasets import Dataset, enable_progress_bar, disable_progress_bar
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets.table import InMemoryTable
import pandas as pd

# Code adapted from this tutorial: https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html

# TODO: Put all this into forcedalignment.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load():
    torch.random.manual_seed(0)
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    return processor, model

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


# Find the most likely path (backtracking)
#
# Once the trellis is generated, we will traverse it following the
# elements with high probability.
#
# We will start from the last label index with the time step of highest
# probability, then, we traverse back in time, picking stay
# ($c_j \rightarrow c_j$) or transition
# ($c_j \rightarrow c_{j+1}$), based on the post-transition
# probability $k_{t, j} p(t+1, c_{j+1})$ or
# $k_{t, j+1} p(t+1, repeat)$.
#
# Transition is done once the label reaches the beginning.
#
# The trellis matrix is used for path-finding, but for the final
# probability of each segment, we take the frame-wise probability from
# emission matrix.
def backtrack(trellis, emission, tokens, transcript, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError(f"Failed to align, {transcript=}")
    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return (
            f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
        )

    @property
    def length(self):
        return self.end - self.start

# Merge the labels
def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def get_timings(processor: Wav2Vec2Processor, model: Wav2Vec2ForCTC, speech_files: list[str], workdir: Path, transcripts: list[str], sub_starts: list[float]):
    worddir = workdir / "words"
    if not worddir.exists():
        worddir.mkdir()

    test_dataset = Dataset(
        InMemoryTable.from_pandas(
            pd.DataFrame(
                {
                    "path": speech_files,
                    "sentence": transcripts,
                }
            )
        )
    )

    # Preprocessing the datasets.
    # We need to read the audio files as arrays
    def speech_file_to_array_fn(batch):
        # TODO: Use filename or just do it in-memory
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
        batch["speech"] = torch.Tensor(speech_array).to(device)
        batch["sentence"] = batch["sentence"].lower()
        with processor.as_target_processor():
            batch["tokenized_sentence"] = processor(batch["sentence"])["input_ids"]
        return batch

    disable_progress_bar()
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    inputs = processor(
        test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True
    )
    enable_progress_bar()

    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu().detach()
    predicted_sentences = processor.batch_decode(predicted_ids)
    res = []
    for i, predicted_sentence in enumerate(predicted_sentences):
        # print("-" * 100)
        # print("Reference:", test_dataset[i]["sentence"])
        # print("Tokens:", list(zip(test_dataset[i]["sentence"])))
        # print("Prediction:", predicted_sentence)

        tokens = test_dataset[i]["tokenized_sentence"]
        emission = logits[i].cpu().detach()

        trellis = get_trellis(emission, tokens)
        # print(f"{trellis=}")

        transcript = transcripts[i]
        transcript = "|".join(transcript.split())
        path = backtrack(trellis, emission, tokens, transcript)
        # print(f"{path=}")

        segments = merge_repeats(path, transcript)
        # for seg in segments:
            # print(seg)

        word_segments = merge_words(segments)
        # for word in word_segments:
            # print(word)

        waveform = test_dataset[i]["speech"]
        sr = 16_000
        def generate_segment(i, j):
            ratio = len(waveform) / (trellis.size(0) - 1)
            word = word_segments[j]
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)
            # filename = str(
            #     worddir / f"{str(word.start+ sub_starts[i]).replace('.', '_')}_{word.label}.wav"
            # )
            # torchaudio.save(filename, torch.Tensor(waveform[x0:x1])[None, :], sr)
            # print(f"{word.label} ({word.score:.2f}): {x0 / sr:.3f} - {x1 / sr:.3f} sec")
            return x0 / sr, x1 / sr
        res.append([generate_segment(i, j) for j in range(len(word_segments))])
    return res
