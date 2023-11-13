[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/7wcZZzeSQk)

# About Spoonfy
**tl;dr**: Spoonfy helps build foreign-language listening skill using AI in a new & unique way. Watch the demo video below to see & hear Spoonfy in action. It's Despacito so it might be a little NSFW:

[![Watch the video](thumbnail.jpg)](https://drive.google.com/file/d/1UFNFPoM7qfBRWUzMbdSTkN-Mny9KONY6/view?usp=sharing)

**What is Spoonfy?** Spoonfy (short for "spoon-feed") is an experimental language learning system made possible by recent advances in state-of-the-art deep learning (AI).

**What does it do?** It helps build listening skill by giving you what I call LitK ("literal karaoke"): Word-by-word translations of every word you hear in the video, as they're being said. Linguists would call this "audio gloss" or something like that (as opposed to literary gloss).

**Why do that?** Each word that appears on-screen becomes its own self-contained mini-lesson teaching you to associate meaning with sound ("What did they say?"), and the context in which the word appears teaches you when it's appropriate to use it ("Are they speaking good Spanish?"). In other words, you'll learn through exposure via [comprehensible input](https://en.wikipedia.org/wiki/Input_hypothesis), just like you did learning your native language.

**Why can't regular subtitles do that?** When subtitles contain a full sentence's translation, you can't easily figure out which sounds you hear belong to which words you see, making it hard to pick up new words just by reading translations. Spoonfy fixes this by presenting the translation in a clever way, as mentioned above.

There's a LOT more to say about the project but that should give you the basic idea.

# About this demo
**tl;dr**: This demo's for you if you're an English speaker trying to learn Spanish and you have an accurately-subtitled Spanish video you wanna learn from.

This demo is the first iteration of the project and is targeted at ML developers & enthusiasts who know what they're doing. It's not ready for end users yet and has some limitations, listed below. But even in its current state, it's already a fully functional learning tool.

Limitations:
- Only supports Spanish (learning)->English (native).
- The video must have Spanish subtitles and Spanish audio. If you have English subtitles, Spoonfy can use them to massively improve the translation quality.
- Subtitles must be in .srt format. Spoonfy will output .ass subtitles since they support karaoke timing.
- The subtitles must be embedded in the input video file in .srt format. Workaround for external subtitles: If the video is `video.mp4`, put the English subtitles in `video/subtitles-eng.srt` and Spanish subtitles in `video/subtitles-spa.srt` (create a `video` folder if needed). If you have embedded or external Spanish subtitles but don't have any English subtitles, create a blank file for the English subs.
- Subtitles must be accurate transcriptions of the audio. I've found (and maybe you've noticed too) that subtitles often simplify what's actually being said.
- Expect translation mistakes, mistimed audio, etc. It is machine translation, after all!
- You'll need a very fast computer to run the model quickly. And about 5 GB of disk space. On my 2017 MacBook Pro (4-core i5) it generates output at around 0.75x speed.
- Slowed-down audio as shown in the demo video requires a video player that supports it. I made the demo video using a hacky modified version of [IINA](https://iina.io) that I don't plan to release anytime soon. To implement yourself, the .ass format subtitles Spoonfy outputs have 3 new commands in their `Dialogue` lines: `{\dialogue}` (turn on any [speech isolators](https://github.com/GregorR/rnnoise-models) which you can apply to the audio using [ffmpeg](https://ffmpeg.org/ffmpeg-filters.html#arnndn)), `{\speed1.0}` (set playback speed) and `{\filler}` (turn off any speech isolators).
- The code is really ugly & buggy. It works, for the most part, but there're lots of improvements to make. I might have to rewrite the whole thing.

These limitations only apply to this demo. As the project grows, they'll be removed.

# Setup
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

    If you're new to Anaconda/Miniconda, it's a package manager like pip except it has packages like entire Python interperters, non-Python programs/libraries, etc. It was created to make Python data scientists' lives easier.
1. Create & activate environment
    ```bash
    conda create --name spoonfy_demo python=3.9
    conda activate spoonfy_demo
    ```
1. Install PyTorch
    ```bash
    # Linux/Windows
    conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch

    # macOS
    conda install pytorch torchaudio -c pytorch
    ```
1. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```
1. Install [FFmpeg](https://ffmpeg.org), make sure `ffprobe` is in your `$PATH`

## Troubleshooting
If you screwed up your environment or need to add/remove packages, save yourself some hassle: Remove the environment and recreate it. Install any new conda/pip packages as you recreate the environment.
```bash
conda env remove --name spoonfy_demo
```

## "Uninstalling"
Linux/macOS: To clean up the downloaded models (~4GB), remove the following:
- `~/.cache/huggingface`
- `~/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`

Windows: Check the Huggingface/PyTorch docs, the paths are different.

# Running the demo
1. Activate Conda environment
    ```bash
    conda activate spoonfy_demo
    ```
1. Run Spoonfy, generate LitK subtitles
    
    If your input subtitles are external files (aka not embedded in the video file), see the workaround in the "Limitations" section.
    ```bash
    python spoonfy.py video.mp4
    ```
    When done, Spoonfy will create a file called (for example) `video.ass`.
1. Play video & LitK subtitles in your favorite video player, e.g. [VLC](https://www.videolan.org/vlc/)
    
    Note that automatic playback slowdown like the demo video has requires a custom video player, see the limitations section for more details.

# About the translation and transcription models
The translation (Spanish->English LitK) model is a finetuned version of Facebook's [M2M100 model](https://huggingface.co/facebook/m2m100_418M) which is available [here](https://huggingface.co/athairus/m2m100_418M_finetuned_litk_es_en). I'll release the finetuned model's dataset & training code "soon". The demo source already has the tokenization code so you already have enough to reproduce the demo's model provided you have your own training data.

The transcription (Audio->Per-word timestamps) model is a pretrained Wav2Vec2 model found [here](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish).

# Code layout
You'll find `main()` in `spoonfy.py`. The various steps done to produce the LitK subtitles are split into different .py files and called from `main()`. See the last group of imports in `spoonfy.py`. Some common data classes are found in `common.py` and a few helper functions are found in `util.py`.

The actual models used for translation & transcription are not found in this repo, see the previous section for links.

# Help wanted!
Come join the project and help the world understand each other like never before! I intend to keep Spoonfy free & open source as I feel something as fundamental as language learning is too important to the world to put behind a paywall.

I'm looking for all kinds of help, from graphic/web designers to ML engineers to app programmers to linguists and anyone & everyone with a good idea/feedback to share.

Join the Discord, come say hi: [![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/7wcZZzeSQk)
