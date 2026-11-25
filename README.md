# reelint

**A linter for short-form video.** Point it at a Reel, Short or TikTok before you post, and it tells you, with timestamps, where viewers are likely to swipe away and how to fix it.

```
$ reelint check draft.mp4
draft.mp4  ·  0:20.0  ·  1080x1920 @ 30fps  ·  score 55/100
  5 cuts  ·  3.0 visual beats per 10s  ·  avg shot 3.33s  ·  -28.3 LUFS

  picture   █      █      █                    █        █             
  sound   ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅    ▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅
  issues  EEE             EEEEEEEEEEEEEEEEEEEEEE        WWWWWWWWWWWWWW
          0:00                                                  0:20.0

  whole video    warning  audio-loudness   Audio is quiet (-28.3 LUFS); it will sound weak next to other posts.
                                           → Normalize to about -14 LUFS, e.g. ffmpeg -af loudnorm=I=-14:TP=-1.5
  0:00.0–0:00.8  error    hook-black-open  Opens on black for 0.8s. The first frame is also the default cover and what people see mid-scroll.
                                           → Cut the fade-in and open on your strongest image.
  0:05.6–0:12.6  error    pacing-static    7.0s with no cut or visual change.
                                           → Add a pattern interrupt: a cut, a zoom, B-roll, an on-screen caption or a graphic.
  0:07.2–0:08.4  warning  audio-dead-air   1.2s of silence.
                                           → Tighten the pause, or cover it with music or a sound effect.
  0:15.6–0:20.0  warning  pacing-static    4.4s with no cut or visual change.
                                           → Add a pattern interrupt: a cut, a zoom, B-roll, an on-screen caption or a graphic.

  2 errors, 3 warnings, 0 info
```

Everything runs on your machine. No uploads, no accounts, no API keys, no scraping.

## Install

You need [FFmpeg](https://ffmpeg.org/download.html) on your PATH and Python 3.10+.

```bash
pip install git+https://github.com/ashlrai/reelint
pip install "reelint[vision] @ git+https://github.com/ashlrai/reelint"   # adds text and face detection for `reelint hook`
```

## Use

```bash
reelint check video.mp4                 # human-readable report
reelint check *.mp4 --format json       # machine-readable, for scripts and AI agents
reelint check video.mp4 --profile tiktok
reelint check video.mp4 --disable audio --disable pacing-static
reelint check video.mp4 --fail-on warning   # non-zero exit for CI / pre-upload hooks
reelint rules                           # list every rule
```

### Profile the hook

```
$ reelint hook clip.mp4 --csv hooks.csv
clip.mp4  ·  hook (first 3s)
  ✓ Opens on a live frame.
  ✓ Text on screen from 0s.
  ✓ First visual change at 1.3s (1 in 3s).
  ✓ Sound from 0s.
  · No face in the first 3s (fine for screen recordings and B-roll).

  Hook text:  "YOU NEED THIS" / "AI TOOL"
```

`reelint hook` measures the opening seconds: when the first text, visual change, sound and face appear, what the on-screen text says, how big it is, and whether the hook is quieter than the rest of the video. Text and face detection need the `vision` extra; without it those checks show as `?`.

It doesn't predict skip rate. Instead, `--csv` keeps a log with one row per video and an empty `skip_rate` column: copy each reel's skip rate from Instagram Insights into it. Re-running keeps what you typed, including any extra columns you add (views, topic, notes). Once you have 20 or more rows you can see which hook features go with lower skip rates for *your* content.

Text detection reads all text in the frame, including text that's part of the scene (a laptop screen, a sign), not only your captions.

Exit codes for `check`: `0` clean, `1` findings at or above `--fail-on` (default `error`), `2` the file couldn't be read.

## Rules

| Rule | Checks |
|---|---|
| `format-aspect` | Vertical 9:16; landscape is an error |
| `format-resolution` | Short side at least 1080px |
| `format-fps` | At least 24fps |
| `format-duration` | Within the platform limit; notes when past 90s |
| `hook-black-open` | Doesn't open on a black frame (it's also the default cover) |
| `hook-static` | Something moves or changes in the first 1.5s |
| `hook-silent-start` | Sound starts within 0.5s |
| `pacing-static` | No stretch longer than 3s (warning) or 6s (error) without a cut or visual change |
| `audio-missing` | Has an audio track |
| `audio-dead-air` | No silences of 1s or more mid-video |
| `audio-loudness` | Integrated loudness between -20 and -9 LUFS, true peak at or below -1 dBTP |
| `ending-black` | Doesn't fade to black (kills the loop) |
| `ending-silent-tail` | Sound runs to the end |

Disable a rule or a whole category (`format`, `hook`, `pacing`, `audio`, `ending`) with `--disable`.

### How "visual change" works

Frames are sampled at 10fps and shrunk to 72x128 pixels. A **beat** is anything a viewer registers as "something changed":

- a **cut**: a large frame-to-frame change;
- a **jump**: a whole-frame change at least 3× the motion in the surrounding 3 seconds, such as a punch-in zoom or a cut between similar shots;
- a **pop**: one region of the frame changes sharply against its usual motion, such as a caption, sticker or logo appearing. The frame is split into a 4×8 grid and the most-changed cell is measured, because a caption is too small to register across the whole frame;
- **drift**: gradual change (a pan, a slow zoom) that has added up to a clear difference since the last beat.

Changes less than 0.5s apart count as one event, so a whip pan is one beat, not ten. `pacing-static` measures the gaps between beats, so punch-ins and caption changes count while a locked-off talking head with no captions doesn't.

## Configure

Put a `reelint.toml` next to your videos (or pass `--config`):

```toml
disable = ["format-duration"]

[thresholds]
static_warn = 4.0     # seconds without a visual beat
static_error = 8.0
hook_window = 2.0
lufs_min = -18.0
```

Every threshold is a field of `Config` in [`src/reelint/config.py`](src/reelint/config.py).

## Use from Python

```python
from reelint import analyze, lint, load_config

config = load_config("reels")
for finding in lint(analyze("draft.mp4", config), config):
    print(finding.start, finding.severity, finding.rule, finding.message)
```

## What the rules are based on

The thresholds are starting points drawn from common short-form editing practice: viewers decide within the first second or two, they expect a visual change every few seconds, platforms normalize loudness to about -14 LUFS, and fades to black end the loop. They are not derived from platform data. Tune them in `reelint.toml`, and if you have retention data that supports different defaults, please open an issue.

## Roadmap

- `caption-safe-zone`: text detection, flagging captions under the platform's UI (username, buttons, description)
- `caption-readability`: how long each caption stays on screen versus how long it takes to read
- Telling captions apart from text that's part of the scene
- `reelint learn`: compare hook features with the skip rates in your log
- Loop-seam scoring: how well the last frame flows into the first
- An MCP server so AI agents can lint and fix their own edits
- Learning thresholds from your own analytics (YouTube Analytics retention curves, Instagram Insights)

## Develop

```bash
pip install -e ".[dev,vision]"
pytest
```

The tests generate their own videos with FFmpeg, so there are no binary fixtures in the repo.

## License

MIT
