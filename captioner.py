"""Everything except the UI: model loading, media handling, prompts, captioning.

One model: ASID-Captioner-7B (Qwen2.5-Omni-7B fine-tune), loaded in 4-bit.
Weights sit flat in ./models/.
"""

import atexit
import gc
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

import torch

# --------------------------------------------------------------------------- #
# paths and model
# --------------------------------------------------------------------------- #

PROJECT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT, "models")

MODEL_NAME = "ASID-Captioner 7B"
MODEL_REPO = "AudioVisual-Caption/ASID-Captioner-7B"
MODEL_NOTES = (
    "Qwen2.5-Omni-7B fine-tuned on ASID-1M with attribute-structured captions: "
    "scene, characters, objects, actions, narrative, speech, camera work and "
    "emotion. Hears the audio track as well as seeing the frames. Loaded in "
    "4-bit, about 8 GB of VRAM."
)

DOWNLOAD_HINT = 'hf download %s --local-dir "%s"' % (MODEL_REPO, MODEL_DIR)


def model_present():
    if not os.path.isdir(MODEL_DIR):
        return False
    return any(f.endswith(".safetensors") for f in os.listdir(MODEL_DIR))


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".wmv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
GIF_EXT = {".gif"}

_FFMPEG_CANDIDATES = [
    os.path.join(PROJECT, "installer_files", "Environments", "captioner",
                 "Library", "bin", "ffmpeg.exe"),
    os.path.join(PROJECT, "bin", "ffmpeg.exe"),
]


def find_ffmpeg():
    for c in _FFMPEG_CANDIDATES:
        if os.path.isfile(c):
            return c
    return shutil.which("ffmpeg")


def _ffmpeg_on_path():
    """audioread shells out to a bare "ffmpeg", so the bundled copy has to be
    on PATH - without this every audio file dies with NotInstalledError."""
    ff = find_ffmpeg()
    if not ff:
        return
    folder = os.path.dirname(os.path.abspath(ff))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if folder and folder not in parts:
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


_ffmpeg_on_path()


def kind(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in GIF_EXT:
        return "gif"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    return None


def scan(folder, recursive=True, wanted=None):
    """Every supported file under `folder`, sorted."""
    found = []
    if not os.path.isdir(folder):
        return found
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for dirpath, dirnames, filenames in walker:
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            if not os.path.isfile(p):
                continue
            k = kind(p)
            if k is None or (wanted and k not in wanted):
                continue
            found.append(p)
    return found


def caption_path(media_path, extension="txt"):
    return os.path.splitext(media_path)[0] + "." + extension.lstrip(".")


def gif_to_mp4(gif_path, fps=12):
    """Convert a GIF to a silent mp4 beside it, so the video path can read it."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. It normally comes with the conda environment - "
            "re-run install.bat, or drop ffmpeg.exe into the bin folder."
        )
    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"
    if os.path.isfile(mp4_path) and os.path.getmtime(mp4_path) >= os.path.getmtime(gif_path):
        return mp4_path
    cmd = [ffmpeg, "-y", "-v", "error", "-i", gif_path,
           "-movflags", "faststart", "-pix_fmt", "yuv420p",
           "-vf", "fps=%d,scale=trunc(iw/2)*2:trunc(ih/2)*2" % fps, mp4_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("GIF conversion failed: %s"
                           % proc.stderr.decode("utf-8", "replace")[:300])
    return mp4_path


_AUDIO_TMP = None


def audio_to_wav(path, sr=16000):
    """Decode any audio - or a video's audio track - to 16 kHz mono wav.

    soundfile cannot open m4a/opus/aac and librosa 1.0 dropped the audioread
    fallback the omni utils rely on, so everything goes through ffmpeg instead.
    Cached per file, wiped when the process exits.
    """
    global _AUDIO_TMP
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found, so the audio cannot be read. Re-run install.bat, "
            "or drop ffmpeg.exe into the bin folder.")

    if _AUDIO_TMP is None:
        _AUDIO_TMP = tempfile.mkdtemp(prefix="asid_audio_")
        atexit.register(shutil.rmtree, _AUDIO_TMP, True)

    full = os.path.abspath(path)
    try:
        stamp = "%s|%d" % (full, os.path.getmtime(full))
    except OSError:
        stamp = full
    out = os.path.join(_AUDIO_TMP,
                       hashlib.md5(stamp.encode("utf-8", "replace")).hexdigest() + ".wav")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out

    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", path, "-vn",
         "-ac", "1", "-ar", str(int(sr)), "-f", "wav", out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError("Could not read the audio of %s: %s"
                           % (os.path.basename(path),
                              proc.stderr.decode("utf-8", "replace")[:200]))
    return out


def _patch_audioread():
    """qwen_omni_utils pulls a video's audio by handing librosa an audioread
    object, which librosa 1.0 no longer accepts. Hand it a wav path instead."""
    try:
        import audioread.ffdec
    except Exception:
        return
    if getattr(audioread.ffdec, "_asid_patched", False):
        return
    audioread.ffdec.FFmpegAudioFile = audio_to_wav
    audioread.ffdec._asid_patched = True


def has_audio_stream(path):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    probe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if not os.path.isfile(probe):
        probe = shutil.which("ffprobe")
    if not probe:
        return False
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

GENERAL_VIDEO = (
    "Caption this video as if you were going to try to generate it with a video "
    "generator. Describe the visual content, how it moves and changes over time, "
    "and the camera work. Also describe the audio, including any speech, music, "
    "or sound effects, and transcribe spoken dialogue verbatim in quotes. Be "
    "decisive by stating things as they are. Do not say things like \"It appears "
    "that\" or \"possibly\". No preamble. Just get to the point."
)

GENERAL_IMAGE = (
    "Caption this image as if you were going to try to generate it with an image "
    "generator. Describe the subject, appearance, pose, setting, lighting, colour "
    "and composition. Be decisive. No preamble. Just get to the point."
)

GENERAL_AUDIO = (
    "Describe this audio. Cover the type of sound, any music (genre, instruments, "
    "tempo, mood), environmental and background sounds, and the character of any "
    "voices (age, gender, tone, accent). Transcribe all speech verbatim in quotes. "
    "No preamble."
)

MINIMAX_T2VA = (
    "Caption this video as a MiniMax text-to-video-audio (T2VA) training prompt. "
    "Watch the video and listen to the audio, then output exactly three fields in "
    "this order, each starting on its own line with these exact field names:\n"
    "\n"
    "short_description: one sentence naming the subject and the main action.\n"
    "\n"
    "integrated_multimodal_description: Start with [Shot 1] (the first shot gets "
    "no timestamp) and state the overall visual style (Live-action, cinematic, "
    "2D-animated, 3D CG, claymation, watercolor, or vintage film) and the initial "
    "framing. Then describe everything visible and audible in chronological order: "
    "subject appearance and position, scene and key props, actions and reactions, "
    "shot changes, speech, and the diegetic sounds that accompany them. Begin each "
    "later shot as \"[Shot 2] At 00:03.500, the camera cuts to ...\" with strictly "
    "increasing cut times. Write camera motion using this vocabulary: zoom in/out, "
    "push in, pull out, pan left/right, truck left/right, tilt up/down, pedestal "
    "up/down, arc shot, tracking shot, static shot, POV, roll clockwise/"
    "counterclockwise, shake slightly/strongly. Give every person who speaks a "
    "stable ID like (S1) or (S2); when a speaker first appears, identify them with "
    "type, age, gender, and voice quality. Transcribe speech verbatim as: The young "
    "woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the "
    "next station.</d> Put legible on-screen text in double quotation marks "
    "verbatim.\n"
    "\n"
    "overall_soundscape: 1-4 sentences summarising ambient sound, physical action "
    "sounds and non-verbal human sounds across the whole video. Do not repeat "
    "dialogue or music here. Use N/A if the video is silent."
)

BOORU_TAGS = (
    "List the contents of this media as comma-separated tags, most important "
    "first. Cover subject, appearance, clothing, pose, action, setting, lighting, "
    "style and mood. Lowercase, no sentences, no preamble."
)

PRESETS = {
    "General - video": GENERAL_VIDEO,
    "General - image": GENERAL_IMAGE,
    "General - audio": GENERAL_AUDIO,
    "MiniMax H3 T2VA": MINIMAX_T2VA,
    "Tags (comma separated)": BOORU_TAGS,
}

AUTO_BY_KIND = {
    "video": GENERAL_VIDEO,
    "gif": GENERAL_VIDEO,
    "image": GENERAL_IMAGE,
    "audio": GENERAL_AUDIO,
}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

SYSTEM = (
    "You are a precise media captioner. You describe exactly what is present in "
    "the media and never invent details."
)

_STATE = {"model": None, "processor": None}


def load(cpu_offload=False, progress=None):
    """Load ASID-Captioner-7B in 4-bit. Cached after the first call."""
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["processor"]

    if not model_present():
        raise RuntimeError(
            "No model in %s\n\nDownload it with:\n\n%s" % (MODEL_DIR, DOWNLOAD_HINT))

    if progress:
        progress("Loading %s ..." % MODEL_NAME)

    from transformers import BitsAndBytesConfig, Qwen2_5OmniProcessor
    try:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration as Cls
    except ImportError:
        from transformers import Qwen2_5OmniForConditionalGeneration as Cls

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {
        "quantization_config": quant,
        "device_map": "auto" if cpu_offload else {"": 0},
        "trust_remote_code": True,
    }
    if cpu_offload:
        kwargs["max_memory"] = {0: "14GiB", "cpu": "64GiB"}

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR)
    model = Cls.from_pretrained(MODEL_DIR, **kwargs)

    # a full omni model still carries the speech talker - we only want text
    if hasattr(model, "disable_talker"):
        try:
            model.disable_talker()
        except Exception:
            pass

    model.eval()
    _STATE["model"] = model
    _STATE["processor"] = processor
    return model, processor


def unload():
    _STATE["model"] = None
    _STATE["processor"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _content_for(path, k, max_pixels):
    if k in ("video", "gif"):
        return [{"type": "video", "video": path, "max_pixels": max_pixels}]
    if k == "image":
        return [{"type": "image", "image": path, "max_pixels": max_pixels}]
    if k == "audio":
        return [{"type": "audio", "audio": audio_to_wav(path)}]
    raise ValueError("unsupported kind: %s" % k)


def _clean(text):
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    for junk in ("assistant\n", "Assistant:", "Caption:"):
        if text.startswith(junk):
            text = text[len(junk):].strip()
    return text


def caption_file(path, prompt, max_new_tokens=512, max_res=512,
                 use_audio=True, cpu_offload=False, progress=None):
    """Caption one file. Returns the caption text."""
    from qwen_omni_utils import process_mm_info
    _patch_audioread()

    model, processor = load(cpu_offload=cpu_offload, progress=progress)

    k = kind(path)
    if k == "gif":
        path = gif_to_mp4(path)
        k = "video"

    audio_in_video = bool(use_audio) and k == "video" and has_audio_stream(path)

    max_pixels = int(max_res) * int(max_res)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user",
         "content": _content_for(path, k, max_pixels) + [{"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=audio_in_video)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True,
                       use_audio_in_video=audio_in_video)
    inputs = inputs.to(model.device)

    gen = dict(max_new_tokens=int(max_new_tokens), do_sample=False,
               use_audio_in_video=audio_in_video)
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)
    except TypeError:
        gen.pop("use_audio_in_video", None)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)

    if isinstance(out, (tuple, list)):
        out = out[0]
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return _clean(processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# prompt maker  -  many references in, one MiniMax H3 prompt out
# --------------------------------------------------------------------------- #
#
# ASID-Captioner is a captioning fine-tune, not an instruction follower. Asked
# for six named fields in one go it either ignores the format completely or
# copies the wording of the instruction back and loops. Asked for exactly one
# thing, with the answer already opened for it, it does the job well. So the
# model is used for what it is good at - looking at each reference and saying
# what is there - and the H3 fields are assembled around those descriptions.

REF_CAPS = {"image": 9, "video": 3, "audio": 3}

REF_TAG = {"image": "Picture", "video": "Video", "audio": "Audio"}


def label_refs(paths):
    """Give every reference its H3 label.

    Returns (refs, dropped) where refs is a list of (label, path, kind) and
    dropped is a list of "why" strings for anything that did not fit.
    """
    refs, dropped, seen = [], [], {"image": 0, "video": 0, "audio": 0}
    for p in paths or []:
        path = p if isinstance(p, str) else getattr(p, "name", None)
        if not path:
            continue
        k = kind(path)
        if k is None:
            dropped.append("%s  -  unsupported type" % os.path.basename(path))
            continue
        if k == "gif":
            k = "video"
        if seen[k] >= REF_CAPS[k]:
            dropped.append("%s  -  over the %d %s limit"
                           % (os.path.basename(path), REF_CAPS[k], k))
            continue
        seen[k] += 1
        refs.append(("<%s %d>" % (REF_TAG[k], seen[k]), path, k))
    # images, then videos, then audio - the order the H3 prompt refers to them in
    order = {"image": 0, "video": 1, "audio": 2}
    refs.sort(key=lambda r: order[r[2]])
    return refs, dropped


# --------------------------------------------------------------------------- #
# stage 1 - look at each reference
# --------------------------------------------------------------------------- #

REF_ROLE_PROMPT = {
    "image": (
        "Describe this image as a reference for a video generator. Cover who or "
        "what the subject is, face and hair, build and age, clothing, any object "
        "or product and its markings, the setting, the lighting and the visual "
        "style. Name colours and materials. Be decisive. One paragraph, no "
        "preamble."
    ),
    "video": (
        "Describe this clip as a reference for a video generator. Cover the "
        "subject, how it moves, the pace of the action, the camera work, the "
        "lighting and the visual style. Then describe the audio: any speech and "
        "what is said, music, and background sound. Be decisive. One paragraph, "
        "no preamble."
    ),
    "audio": (
        "Describe this audio as a reference for a video generator. Cover any "
        "music - instruments, tempo, mood - the background and environmental "
        "sound, and the character of any voice: age, gender, accent, pace and "
        "tone. Transcribe speech verbatim in quotes. Be decisive. One paragraph, "
        "no preamble."
    ),
}

# The fine-tune opens almost everything with "At 0s, the video opens with a
# static shot of ...", even for a still image. Trim that off the front.
_LEAD = re.compile(
    r"^\s*(at\s+\d+(?:\.\d+)?\s*s(?:econds)?\s*,\s*)?"
    r"(the\s+(video|image|clip|audio)\s+(opens\s+(with|on|in)|begins\s+(with|on)|"
    r"shows|features|depicts|is)\s*)?"
    r"((a|an)\s+(?:[\w-]+\s+){0,3}shot\s+(?:of\s+)?)?",
    re.I)


def _strip_lead(text):
    text = (text or "").strip()
    trimmed = _LEAD.sub("", text, count=1).strip()
    return (trimmed[:1].upper() + trimmed[1:]) if trimmed else text


# Given a still image the fine-tune still invents "At 5s, the square continues
# ...". A reference note only needs the look, so keep the part before that.
_TIMELINE = re.compile(r"\s*at\s+\d+(?:\.\d+)?\s*s(?:econds)?\s*,\s+", re.I)


# How long a note may run, and how much the model gets to say. A clip has to
# fit its motion AND its audio in here - cut it short and the note is nothing
# but the opening scenery, which is the one thing the clip was not needed for.
NOTE_LIMIT = {"image": 900, "video": 1250, "audio": 800}
NOTE_TOKENS = {"image": 260, "video": 420, "audio": 240}


def _trim_note(text, k="video"):
    """Trim a reference note. The per-second timeline is invented for a still
    image, so it goes; for a clip it is the motion, so it stays."""
    text = (text or "").strip()
    if k == "image":
        first = _TIMELINE.split(text)[0].strip()
        if len(first) >= 40:      # shorter than that and the note is all timeline
            text = first
    limit = NOTE_LIMIT.get(k, 900)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return cut[:dot + 1].strip() if dot > limit // 3 else cut.strip() + " ..."


def describe_refs(refs, max_res=448, video_fps=1.0, use_audio=True,
                  cpu_offload=False, on_step=None):
    """Caption every reference. Returns [(label, path, kind, description)]."""
    notes = []
    for i, (label, path, k) in enumerate(refs):
        if on_step:
            on_step(i, len(refs), label, os.path.basename(path))
        # one still is cheap next to a whole clip, so it gets the sharper look -
        # this is where hair, eyes, clothing and props are actually read
        res = min(int(max_res) * 3 // 2, 768) if k == "image" else max_res
        try:
            cap = caption_file(path, REF_ROLE_PROMPT[k],
                               max_new_tokens=NOTE_TOKENS[k], max_res=res,
                               use_audio=use_audio, cpu_offload=cpu_offload)
            notes.append((label, path, k, _trim_note(_strip_lead(cap), k)))
        except Exception as exc:
            notes.append((label, path, k, "could not be read: %s" % exc))
    return notes


# --------------------------------------------------------------------------- #
# stage 2 - narrow, single-answer asks
# --------------------------------------------------------------------------- #

BUILDER_SYSTEM = (
    "You write shot descriptions for a video generator. You keep to what the "
    "reference notes actually say and you never invent details that contradict "
    "them."
)


def _ask(model, processor, user, prefill="", max_new_tokens=500,
         system=BUILDER_SYSTEM):
    """One text-only question, with the answer opened for the model."""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    if prefill:
        text += prefill
    inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                             do_sample=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    answer = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return _clean(answer)


def _notes_block(notes):
    return "\n".join("%s - %s" % (label, desc) for label, _p, _k, desc in notes)


def _intent_block(intent):
    return intent.strip() if (intent or "").strip() else \
        "Not stated - follow the references."


# The instruction the user can edit. No worked example in here: shown one, the
# model copies its wording into the answer instead of describing the media.
SHOT_INSTRUCTION = (
    "Write it shot by shot, in playback order.\n"
    "\n"
    "Open with the visual style - live-action, cinematic, 2D animation, 3D CG, "
    "claymation, watercolour or vintage film - and the opening framing. Give each "
    "shot the subject's state, the action, the environment and one camera move. "
    "Mark a new shot with its start time in the form [Shot 2] At 00:03.500, with "
    "strictly increasing times.\n"
    "\n"
    "Use only these camera moves: zoom in, zoom out, push in, pull out, pan left, "
    "pan right, truck left, truck right, tilt up, tilt down, pedestal up, pedestal "
    "down, arc shot, tracking shot, static shot, POV, roll clockwise, roll "
    "counterclockwise, shake slightly, shake strongly.\n"
    "\n"
    "Refer to the references by their labels. Wrap every spoken line in <d>[English] "
    "...</d> and put on-screen text in double quotes. Present tense, decisive. No "
    "preamble and no commentary. Stop between two hundred and three hundred words."
)

PROSE_INSTRUCTION = (
    "Describe the video to make, which is the one under \"The video to make\" "
    "above. Do not describe the references one by one - they are only source "
    "material. A clip gives the action, timing, camera and place; a still gives a "
    "person's face, hair, body and clothes and nothing else about it; a sound "
    "reference gives the music and the voice.\n"
    "\n"
    "Write one MiniMax H3 prompt as flowing prose, 120-200 words, no field names "
    "and no bullet points.\n"
    "\n"
    "Cover the visual style, what each reference controls - naming it by its label "
    "- the opening composition, the action in time order, one camera move per shot, "
    "the spoken line, the scene sounds and any music, and the closing composition.\n"
    "\n"
    "Use only these camera moves: zoom in, zoom out, push in, pull out, pan left, "
    "pan right, truck left, truck right, tilt up, tilt down, pedestal up, pedestal "
    "down, arc shot, tracking shot, static shot, POV, roll clockwise, roll "
    "counterclockwise, shake slightly, shake strongly.\n"
    "\n"
    "Present tense, decisive. No preamble and no commentary."
)

STRUCTURED = "MiniMax H3 - reference to video (six fields)"
PROSE = "MiniMax H3 - simple prose"

BUILDERS = {STRUCTURED: SHOT_INSTRUCTION, PROSE: PROSE_INSTRUCTION}


# --------------------------------------------------------------------------- #
# stage 3 - assemble the fields
# --------------------------------------------------------------------------- #

# What a reference of each kind is normally there to do, in H3's own terms.
# a leading "00:00s," or "At 0s," - only ever stripped when a comma follows, so
# it cannot eat the description itself
_OPEN_STAMP = re.compile(
    r"^(?:at\s+)?\d{1,2}(?::\d{2}){0,2}(?:\.\d{1,3})?\s*s?(?:econds)?\s*,\s*", re.I)

RETENTION = {"image": "fully_preserved",
             "video": "attribute_transfer",
             "audio": "reference"}

REF_JOB = {"image": "supplies identity and look",
           "video": "supplies motion, pacing and camera style",
           "audio": "supplies the sound and voice character"}


def _subject_definitions(notes):
    swap = _has_still(notes)
    lines, n = [], 0
    for label, _path, k, desc in notes:
        if k == "image":
            n += 1
            lines.append("<Subject %d> is the subject of %s: %s"
                         % (n, label, _person_only(desc)))
        elif k == "video":
            # with a still in play the clip's own performer is being replaced,
            # so she is named by role here rather than by what she is wearing
            body = swap_subject(desc, "the performer") if swap else desc
            lines.append("%s is a motion and style reference: %s" % (label, body))
        else:
            lines.append("%s is a sound reference: %s" % (label, desc))
    return "\n".join(lines) if lines else "No references supplied."


def _summary(notes, intent):
    jobs = "  ".join("%s %s." % (label, REF_JOB[k]) for label, _p, k, _d in notes)
    head = "[reference generation] " if notes else "[text to video] "
    return (head + _intent_block(intent) + ("  " + jobs if jobs else "")).strip()


def _retention(notes):
    if not notes:
        return "No references supplied."
    return "\n".join("%s  %s" % (label, RETENTION[k]) for label, _p, k, _d in notes)


# Used when there is no clip to rewrite and the shots have to be composed.
ROLE_RULES = (
    "Describe the video to make, not the references one by one - they are only "
    "source material. A still gives a person's face, hair, body and clothes, or "
    "an object; take those and drop everything else about it, especially a plain "
    "studio background, studio lighting and the pose it happens to be standing "
    "in. A sound reference gives the music and the voice."
)


def _clip_note(notes):
    """The clip carries the action, so it is the spine when there is one."""
    for label, _p, k, desc in notes:
        if k == "video":
            return label, desc
    return None


def _has_still(notes):
    return any(k == "image" for _lb, _p, k, _d in notes)


# The clip's own performer, as this fine-tune always words it.
_WEARING = re.compile(r",?\s*\b(?:wearing|dressed in|clad in)\b[^.;]*", re.I)
_PERSON = re.compile(
    r"\b(?:a|an|the)\s+(?:young\s+|older\s+|middle-aged\s+|little\s+)?"
    r"(?:wo)?m(?:a|e)n\b|\b(?:a|an|the)\s+(?:young\s+|little\s+)?"
    r"(?:girl|boy|lady|guy|person|teenager|child)\b", re.I)
_WITH_LOOK = re.compile(
    r"\s+with\s+(?:(?:long|short|shoulder-length|dark|light|blonde|blond|black|"
    r"brown|red|grey|gray|curly|straight|wavy|tied-back)\s+){1,3}"
    r"(?:hair|eyes|curls|ponytail|pigtails|braids?|beard|moustache|mustache)\b",
    re.I)


# A still contributes a person, not the studio it was shot in.
_STUDIO_CLAUSE = re.compile(
    r",?\s*(?:against|on|in front of|over)\s+(?:a|an|the)\s+"
    r"(?:plain|solid|simple|neutral|flat)?\s*"
    r"(?:white|grey|gray|black|transparent|light)\s+background", re.I)
_STUDIO_SENT = re.compile(
    r"the scene is minimalistic|no additional decor|no additional furniture|"
    r"keeping (?:the )?focus entirely|the background is (?:a )?(?:plain|solid|white)|"
    r"there (?:is|are) no (?:other )?(?:decor|background|furniture)", re.I)


def _person_only(text):
    """The look a still contributes, with its studio stripped off."""
    text = _STUDIO_CLAUSE.sub("", text or "")
    kept = [s for s in re.split(r"(?<=\.)\s+", text) if not _STUDIO_SENT.search(s)]
    text = _LEAD.sub("", " ".join(kept).strip(), count=1).strip()
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return (text[:1].upper() + text[1:]) if text else text


# Which sentences of a clip note are its music and which are its ambience.
_MUSIC_WORDS = re.compile(
    r"\b(music|track|song|melody|melodic|bassline|beat|tempo|synth\w*|piano|guitar|"
    r"drums?|instrumental|score|sings?|singing|vocals?|lyrics)\b", re.I)
_AMBIENT_WORDS = re.compile(
    r"\b(wind|rain|thunder|footsteps?|traffic|birds?|ambien\w+|rustl\w+|hum|engine|"
    r"crowd|chatter|waves?|breathing|laughter|click\w*|creak\w*|rumble|echo|"
    r"sound effects?|background (?:noise|sound))\b", re.I)


def _audio_sentences(text):
    """(ambience, music) pulled straight out of the clip note."""
    sents = [s for s in re.split(r"(?<=\.)\s+", text or "")
             if not _OPEN_STAMP.match(s.strip())]
    music = [s.strip() for s in sents if _MUSIC_WORDS.search(s)]
    amb = [s.strip() for s in sents
           if _AMBIENT_WORDS.search(s) and not _MUSIC_WORDS.search(s)]
    return " ".join(amb).strip(), " ".join(music).strip()


def swap_subject(text, label="<Subject 1>"):
    """Point the clip's action at the new subject.

    Asked to do this itself the model copies the appearance back verbatim every
    time - three phrasings of the request, three verbatim copies - so the
    substitution is done here instead, where it is reliable and reviewable.
    """
    text = _WEARING.sub("", text)
    text = _WITH_LOOK.sub("", text)
    text = _PERSON.sub(label, text)
    text = re.sub(r"(%s)(\s+\1)+" % re.escape(label), label, text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _context_for(notes, intent, template, fmt):
    """The question put to the model.

    With a clip in hand this is a rewrite - here is what the clip does, say it
    again with one thing changed - which the fine-tune handles. Without one it
    has to compose from stills, which it does far less well.

    The swap itself is never asked for in prose. H3 already has a mechanism for
    it: the shots call her <Subject 1> and subject_definitions says who that is.
    Asking for a label substitution instead of an appearance rewrite is a much
    smaller job, and it is the correct H3 form either way.

    The form rules go first and the material last - put them the other way round
    and the model carries straight on writing the rules back out.
    """
    instruction = template or BUILDERS[fmt]
    clip = _clip_note(notes)

    if clip is None:
        return ("%s\n\n%s\n\nReference notes - only how each reference looks and "
                "sounds:\n%s\n\nThe video to make:\n%s\n\nNow write it."
                % (ROLE_RULES, instruction,
                   _notes_block(notes) or "None.", _intent_block(intent)))

    label, clip_desc = clip
    swap = ("The person doing all of it is <Subject 1>. Do not describe how she "
            "looks anywhere - just call her <Subject 1>. Her appearance is "
            "already defined elsewhere, and the still she comes from contributes "
            "nothing else: no plain or white studio background, no studio "
            "lighting, no standing pose.\n"
            if _has_still(notes) else "")

    return ("%s\n"
            "\nThis is what the reference clip %s does, from beginning to end:\n%s\n"
            "\nWrite that out again as the video to make, changing only this:\n"
            "%s%s\n"
            "\nEverything else stays exactly as the clip has it - every action, "
            "every time, every camera move and every sound.\n"
            "\nNow write it."
            % (instruction, label, clip_desc, swap, _intent_block(intent)))


def _sound_context(notes, intent):
    """Small, self-contained context for the two sound questions."""
    clip = _clip_note(notes)
    if clip:
        return ("This is what the video to make contains:\n%s\n"
                "\nWith this change:\n%s\n\n" % (clip[1], _intent_block(intent)))
    return ("Reference notes:\n%s\n\nThe video to make:\n%s\n\n"
            % (_notes_block(notes) or "None.", _intent_block(intent)))


def build_prompt(paths, intent, template, fmt=STRUCTURED, max_new_tokens=600,
                 max_res=384, video_fps=1.0, use_audio=True, cpu_offload=False,
                 on_step=None):
    """Read a mixed pile of references and write one generation prompt.

    Returns (prompt_text, notes, dropped).
    """
    refs, dropped = label_refs(paths)
    intent = (intent or "").strip()
    if not refs and not intent:
        raise ValueError("Add at least one reference file, or say what you want.")

    model, processor = load(cpu_offload=cpu_offload)

    notes = describe_refs(refs, max_res=max_res, video_fps=video_fps,
                          use_audio=use_audio, cpu_offload=cpu_offload,
                          on_step=on_step)

    context = _context_for(notes, intent, template, fmt)
    if on_step:
        on_step(len(refs), len(refs), "", "writing the prompt")

    if fmt == PROSE:
        return _clean(_ask(model, processor, context,
                           max_new_tokens=max_new_tokens)), notes, dropped

    # "[Shot 1] The video opens with " lands on the phrasing the fine-tune
    # already uses, minus the "At 0s," - H3 gives the first shot no timestamp.
    opener = "[Shot 1] The video opens with "
    shots = _ask(model, processor, context, prefill=opener,
                 max_new_tokens=max_new_tokens)
    # it likes to carry on with "00:00s, ..." - shot 1 takes no timestamp
    shots = _OPEN_STAMP.sub("", shots.strip().lstrip(",. "), count=1)
    shots = opener + shots
    # "The video opens with showing a rooftop" - it joins onto the opener badly
    shots = re.sub(r"\bopens with (?:showing|shows|depicting|featuring)\s+",
                   "opens with ", shots, count=1, flags=re.I)
    if _clip_note(notes) and _has_still(notes):
        shots = swap_subject(shots)

    # A clip already told us what it sounds like, so take it from the note
    # rather than asking again - asked cold the model just answers N/A.
    clip = _clip_note(notes)
    sound, music = _audio_sentences(clip[1]) if clip else ("", "")

    # a clip that named its music but no ambience simply has none worth naming;
    # asking anyway just gets "No sound is described."
    if clip and music and not sound:
        sound = "N/A"

    if not sound or not music:
        sound_ctx = _sound_context(notes, intent)
        if not sound:
            sound = _ask(model, processor, sound_ctx +
                         "Describe its sound in 1-3 sentences: the ambience and "
                         "the physical action sounds - wind, rain, footsteps, "
                         "impacts, breathing. No dialogue and no music. Answer N/A "
                         "only if it is genuinely silent. No preamble.",
                         max_new_tokens=140)
        if not music:
            music = _ask(model, processor, sound_ctx +
                         "Describe its music in one sentence: the instruments, the "
                         "tempo and the dynamics. No mood words. Answer N/A only if "
                         "there is genuinely no music. No preamble.",
                         max_new_tokens=100)

    return ("subject_definitions:\n%s\n\n"
            "summary:\n%s\n\n"
            "retention_analysis:\n%s\n\n"
            "detailed_description:\n%s\n\n"
            "overall_soundscape:\n%s\n\n"
            "non_diegetic_music:\n%s"
            % (_subject_definitions(notes), _summary(notes, intent),
               _retention(notes), shots.strip(),
               sound.strip() or "N/A", music.strip() or "N/A")), notes, dropped
