from io import TextIOWrapper
from common import *
import syltippy
import ffmpeg
from pathlib import Path


def get_row(
    subtitle: list[WordGroup],
    is_filler: bool,
    desired_rate: float,
    min_video_rate: float,
    curr_color: int,
    colors: list[str],
    colors_bright: list[str],
) -> tuple[str, int, float]:  # row, curr_color, rate
    """
    Generates a single Dialogue event's Text field for a .ass file. This is the main part of a single on-screen subtitle.
    Also returns next color to use along with a playback rate (for use in said Dialogue event's Effect field).
    """
    res = []

    sub_start = subtitle[0].start
    sub_end = subtitle[-1].end

    # Get syllables/sec
    if not is_filler:
        syllables = sum(len(syltippy.syllabize(group.source)[0]) for group in subtitle)
        sub_len = sub_end - sub_start
        if sub_len == 0:
            sub_len = 1  # ?
        freq = syllables / sub_len

        # Scale that 1 syllable/sec to the user's desired rate
        freq /= desired_rate

        # Clamp to 1x speed max. No speeding up the video!
        freq = max(freq, 1.0)

        # Clamp to whatever minimum speed the user desires
        freq = min(freq, 1.0 / min_video_rate)

        # Scale factor
        rate = 1.0 / freq
        res.append("{" "\\dialogue" "}")
    else:
        rate = 1.0
        res.append("{" "\\filler" "}")
    res.append("{" f"\\speed{rate}" "}")

    for i in range(len(subtitle)):
        group = subtitle[i]
        curr_color = (curr_color + 1) % len(colors)

        last_group = i == len(subtitle) - 1
        next_group = subtitle[i + 1] if not last_group else None

        # Blank entry? Make the next group's start time this blank entry's start time... if there is a next one
        if not group.target:
            if not last_group:
                next_group.start = group.start
            continue

        # Within an event, .ass timestamps are in ms and are relative to the start of the event (aka subtitle)
        start = (group.start - sub_start) * 1000
        end = (group.end - sub_start) * 1000

        # Animation duration. This does *not* include the gap b/w this group's end and the
        # next group's start
        dur = end - start

        # Karaoke duration. This should include the gap b/w this group & the next one
        # If it's the last group, there is no "next" one so just use the last group's end time
        next_start = (next_group.start - sub_start) * 1000 if not last_group else end

        # Karaoke duration must be in hundredths of a second...
        kdur = (next_start - start) / 10

        # Write karaoke template and word group
        res.append(
            "{"
            # Reset styles for remainder of line (everything prior is kept as-is)
            f"\\r"
            # Karaoke duration.
            f"\\k{round(kdur):d}"
            # Set initial color now
            f"\\1c&H{colors_bright[curr_color]}&"
            # Change color gradually
            # Text: Orange (Note: hex is in BGR order)
            f"\\t({round(start):d},{round(end):d},\\1c&H{colors[curr_color]}&)"
            # Border: Opaque
            f"\\t({round(start):d},{round(end):d},\\3a&H00)"
            # Shadow: Opaque
            f"\\t({round(start):d},{round(end):d},\\4a&H00)"
            # TODO: Position word
            # Grow word group size during first 1/3 of karaoke duration
            f"\\t({round(start):d},{round(start+dur*0.3):d},\\fscy120)"
            # Shrink back to normal during remaining 2/3
            f"\\t({round(start+dur*0.3):d},{round(end):d},\\fscy100)"
            "}"
            # Write group
            f"{group.target}"
        )

    return " ".join(res), curr_color, rate


def get_ts(ts: float) -> str:
    """Returns timestamp in HH:MM:SS.HH format"""
    hours, remainder = divmod(ts, 3600)
    minutes, seconds = divmod(remainder, 60)
    hundredths = (seconds * 100) % 100
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(hundredths):02}"


def write_sub(
    af: TextIOWrapper,
    sub: list[WordGroup],
    is_filler: bool,
    desired_rate: float,
    min_video_rate: float,
    curr_color: int,
    colors: list[str],
    colors_bright: list[str],
) -> int:
    """Writes a Dialogue event line to the given .ass file using the provided subtitle (a list of WordGroups)."""
    start, end = get_ts(sub[0].start), get_ts(sub[-1].end)
    row, curr_color, rate = get_row(
        sub, is_filler, desired_rate, min_video_rate, curr_color, colors, colors_bright
    )

    af.write("Dialogue: 0,")
    af.write(start)
    af.write(",")
    af.write(end)
    af.write(",Default,,0,0,0,")
    af.write(
        f"Speed;{rate},"
    )  # Custom effect field. Requires modified video player to use
    af.write(row)
    af.write("\n")

    return curr_color


def generate_ass(
    video_fn: Path,
    groups: list[WordGroup],
    desired_rate: float,
    min_video_rate: float,
):
    """
    Generates a .ass file using the given list of WordGroups, automatically grouping the given words into
    appropriately-sized subtitles.
    """
    ass_fn = video_fn.parent / f"{video_fn.stem}.ass"

    # Read input video, get length, fps, sample rate
    probe = ffmpeg.probe(
        str(video_fn),
        # cmd="/path/to/ffmpeg-5.0-amd64-static/ffprobe",  # FIXME: Why did I need FFmpeg 5.0 here? I forgot
    )
    video_len = float(probe["format"]["duration"])
    video_fps = 0
    video_sr = 0
    for stream in probe["streams"]:
        codec_type = stream["codec_type"]
        if codec_type == "video":
            fps_str = stream["avg_frame_rate"]
            fps_frac = fps_str.split("/")
            video_fps = float(fps_frac[0])
            if len(fps_frac) > 1:
                video_fps = video_fps / float(fps_frac[1])
        if codec_type == "audio":
            video_sr = float(stream["sample_rate"])

    # Group word list into subtitles
    # The algorithm primarily looks at time b/w words & sentence length to determine when to make another subtitle
    # It'll also add invisible filler words to ensure each subtitle's last word stays on-screen for a reasonable amount of time
    subs: list[list[WordGroup]] = [[]]
    # TODO: Make these parameters
    # Max number of seconds b/w adjacent words. If exceeded, a new subtitle is made unconditionally
    cutoff_time = 1
    # Max number of words per subtitle before we start cutting off by sentence
    cutoff_words = 5
    # Max number of words before we force a new subtitle unconditionally (for those run-on sentences...)
    cutoff_words_forced = 15
    # How long to keep the end of the current subtitle on-screen. Especially useful when single-word subtitles are very short
    filler_word_time = 1
    for group in groups:
        # If there's a prev word to compare with (aka not the very 1st word in video) and either
        # 1. This word (candidte for next subtitle) & the previous word are more than threshold seconds apart, or
        # 2. We reached the number of cutoff words and the last group indicated it was the end of its sentence
        # (making it a natural splitting point, note the LHS counts target words & the RHS is derived from source words), or
        # 3. We reached the number of forced cutoff words
        # TODO: Add scene change detection (requires some CV model), split by clause (requires some NLP model)
        # TODO: Maybe the two could be unified into a model all on its own?
        n_words = sum(len(group.target.split()) for group in subs[-1])
        if subs[0] and (
            group.start - subs[-1][-1].end > cutoff_time
            or (n_words > cutoff_words and subs[-1][-1].end_of_sentence)
            or n_words > cutoff_words_forced
        ):
            # Now that we're at the end of a subtitle we can add a filler word to give the user a moment to digest the
            # last word as it'd likely disappear very quickly otherwise
            # Calculate filler word start/end using prev word's end time and curr word's start time
            # We're guaranteed to have a previous word so we can safely do blind lookbehind
            prev_end = subs[-1][-1].end  # Last word in curr subtitle
            next_start = group.start  # Next word we've yet to append

            curr_filler_time = min(  # Do not let this filler overlap w/ next word
                filler_word_time, next_start - prev_end
            )
            subs[-1].append(
                WordGroup(
                    "",
                    target="",
                    start=prev_end,
                    end=prev_end + curr_filler_time,
                )
            )

            # Start a new subtitle
            subs.append([])
        subs[-1].append(group)

    # # Rewind each subtitle start by a small amount of time (to work around the delay when activating rubberband/voice isolator RNN effects)
    # prev_sub = None
    # for sub in subs:
    #     sub[0].start = sub[0].start - 0.25
    #     # Handle overlaps (underlaps? :) )
    #     if prev_sub and prev_sub[-1].end > sub[0].start:
    #         prev_sub[-1].end = sub[0].start
    #     prev_sub = sub

    # Write .ass file
    with open(ass_fn, "w") as af:
        # Write header lines
        af.write("[Script Info]\n")
        af.write("PlayResX: 1920\n")
        af.write("PlayResY: 1080\n")
        af.write("WrapStyle: 0\n")
        af.write("ScaledBorderAndShadow: yes\n")
        af.write("Collisions: Normal\n")
        af.write("\n")
        af.write("[Aegisub Project Garbage]\n")
        af.write("Audio File: {}\n".format(video_fn))
        af.write("Active Line: {}\n".format(len(subs)))
        af.write("\n")
        af.write(
            """
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default-furigana,Arial,24,&H00FFFFFF,&HFF808080,&HFF000000,&HFF000000,0,0,0,0,100,100,0,0,1,1,1,1,10,10,10,1
Style: Default,Arial,48,&H00FFFFFF,&HFF808080,&HFF000000,&HFF000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
"""
        )
        af.write("\n")
        af.write("[Events]\n")
        af.write(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        # Set up colors
        # RGBY: https://paletton.com/#uid=75z1j0kvcB5fmARn7ywHru2OHpk
        colors = ["F54459", "FFD447", "4C46B8", "5DDA3C"]
        curr_color = 0
        colors_bright = colors.copy()
        brighten = 0x80
        # .ass subtitles use BGR color for some reason...
        for i, color in enumerate(colors):
            r, g, b = color[0:2], color[2:4], color[4:6]
            r, g, b = int(r, base=16), int(g, base=16), int(b, base=16)
            colors[i] = hex(b)[2:] + hex(g)[2:] + hex(r)[2:]
            r, g, b = (
                min(r + brighten, 255),
                min(g + brighten, 255),
                min(b + brighten, 255),
            )
            colors_bright[i] = hex(b)[2:] + hex(g)[2:] + hex(r)[2:]

        # Write each subtitle line
        for i, sub in enumerate(subs):
            # print(" ".join([group.target for group in sub]))

            # Add initial filler segment
            if sub == subs[0]:
                write_sub(
                    af,
                    [WordGroup("", start=0, end=sub[0].start)],
                    True,
                    desired_rate,
                    min_video_rate,
                    curr_color,
                    colors,
                    colors_bright,
                )

            # Write the actual subtitle
            curr_color = write_sub(
                af,
                sub,
                False,
                desired_rate,
                min_video_rate,
                curr_color,
                colors,
                colors_bright,
            )

            # Add filler segment b/w subs or to the end of the video
            start, end = sub[-1].end, 0
            if sub != subs[-1]:
                end = subs[i + 1][0].start  # First word in next subtitle
            else:  # End of video
                end = video_len
            write_sub(
                af,
                [WordGroup("", start=start, end=end)],
                True,
                desired_rate,
                min_video_rate,
                curr_color,
                colors,
                colors_bright,
            )
