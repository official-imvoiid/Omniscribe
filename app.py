"""ASID Captioner - web UI."""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

import captioner as C

KIND_LABELS = {"Video": "video", "GIF": "gif", "Image": "image", "Audio": "audio"}
DEFAULT_KINDS = list(KIND_LABELS.keys())
AUTO = "Auto (by file type)"
PRESET_NAMES = [AUTO] + list(C.PRESETS.keys())
BUILDER_NAMES = list(C.BUILDERS.keys())

# Auto picks a prompt per file, so show all three rather than an empty box.
AUTO_PREVIEW = "\n\n".join([
    "VIDEO and GIF\n" + C.GENERAL_VIDEO,
    "IMAGE\n" + C.GENERAL_IMAGE,
    "AUDIO\n" + C.GENERAL_AUDIO,
])


def _kinds(selected):
    return {KIND_LABELS[s] for s in selected if s in KIND_LABELS}


def _prompt_for(kind, preset, prompt_text):
    if preset == AUTO:
        return C.AUTO_BY_KIND.get(kind, C.GENERAL_VIDEO)
    return prompt_text or C.PRESETS.get(preset, C.GENERAL_VIDEO)


# --------------------------------------------------------------------------- #
# terminal banner
# --------------------------------------------------------------------------- #

def _short(path):
    """Paths relative to the project folder - the absolute ones wrap the console."""
    if not path:
        return path
    try:
        rel = os.path.relpath(path, C.PROJECT)
        if not rel.startswith(".."):
            return ".\\" + rel if os.sep == "\\" else "./" + rel
    except Exception:
        pass
    return path


def banner(port):
    def row(k, v):
        return "  %-10s %s" % (k, v)

    lines = ["", "  ASID Captioner", "  " + "-" * 54]

    if C.model_present():
        shards = [f for f in os.listdir(C.MODEL_DIR) if f.endswith(".safetensors")]
        gb = sum(os.path.getsize(os.path.join(C.MODEL_DIR, f)) for f in shards) / 1e9
        lines.append(row("Model", "ASID-Captioner 7B   4-bit NF4"))
        lines.append(row("Weights", "%s   %d shards, %.1f GB"
                         % (_short(C.MODEL_DIR), len(shards), gb)))
    else:
        lines.append(row("Model", "NOT FOUND in %s" % _short(C.MODEL_DIR)))

    try:
        import torch
        if torch.cuda.is_available():
            lines.append(row("Device", "%s   %.0f GB"
                             % (torch.cuda.get_device_name(0),
                                torch.cuda.get_device_properties(0).total_memory / 1e9)))
        else:
            lines.append(row("Device", "CPU ONLY - captioning will be very slow"))
        lines.append(row("Torch", torch.__version__))
    except Exception as exc:
        lines.append(row("Torch", "not available: %s" % exc))

    ff = C.find_ffmpeg()
    if ff and os.path.abspath(ff).startswith(os.path.abspath(C.PROJECT)):
        lines.append(row("ffmpeg", "bundled with the environment"))
    else:
        lines.append(row("ffmpeg", ff if ff else "NOT FOUND - GIF and audio disabled"))

    lines.append("  " + "-" * 54)
    lines.append(row("Web UI", "http://127.0.0.1:%d" % port))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #

def _files_box(text):
    """Scan lists files; captioning logs progress. Same box, honest label."""
    return gr.update(value=text, label="Files")


def do_scan(folder, recursive, selected_kinds):
    folder = (folder or "").strip().strip('"')
    if not folder:
        return _files_box("Enter a folder path first.")
    if not os.path.isdir(folder):
        return _files_box("Not a folder:  %s" % folder)
    files = C.scan(folder, recursive, _kinds(selected_kinds))
    if not files:
        return _files_box("No supported files in  %s" % folder)

    counts = {}
    for f in files:
        counts[C.kind(f)] = counts.get(C.kind(f), 0) + 1
    summary = "   ".join("%d %s" % (v, k) for k, v in sorted(counts.items()))
    out = ["%d files          %s" % (len(files), summary), ""]
    out += ["  " + os.path.relpath(f, folder) for f in files[:18]]
    if len(files) > 18:
        out.append("  ... and %d more" % (len(files) - 18))
    return _files_box("\n".join(out))


def do_caption(folder, recursive, selected_kinds, preset, prompt_text,
               max_new_tokens, max_res, use_audio, cpu_offload, skip_existing,
               caption_ext, progress=gr.Progress()):
    folder = (folder or "").strip().strip('"')
    log = []

    def emit(line):
        log.append(line)
        print(line, flush=True)
        return gr.update(value="\n".join(log), label="Progress")

    if not os.path.isdir(folder):
        yield emit("Not a folder:  %s" % folder)
        return

    files = C.scan(folder, recursive, _kinds(selected_kinds))
    if not files:
        yield emit("No supported files found.")
        return

    if C.find_ffmpeg() is None:
        yield emit("!  ffmpeg not found - GIF conversion and audio detection are off.\n")

    yield emit("%d files found. Loading model, first run takes a minute ...\n" % len(files))

    done = skipped = failed = 0
    for i, path in enumerate(files):
        rel = os.path.relpath(path, folder)
        progress((i, len(files)), desc=rel)

        out_path = C.caption_path(path, caption_ext)
        if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            yield emit("%4d/%-4d  skip   %s" % (i + 1, len(files), rel))
            continue

        k = C.kind(path)
        try:
            caption = C.caption_file(
                path, _prompt_for(k, preset, prompt_text),
                max_new_tokens=max_new_tokens, max_res=max_res,
                use_audio=use_audio, cpu_offload=cpu_offload)

            if k == "gif":
                mp4 = os.path.splitext(path)[0] + ".mp4"
                with open(C.caption_path(mp4, caption_ext), "w", encoding="utf-8") as fh:
                    fh.write(caption)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(caption)

            done += 1
            yield emit("%4d/%-4d  ok     %s\n               %s"
                       % (i + 1, len(files), rel, caption.replace("\n", " ")[:96]))
        except Exception as exc:
            failed += 1
            yield emit("%4d/%-4d  FAIL   %s\n               %s"
                       % (i + 1, len(files), rel, str(exc).strip()[:260]))
            if failed == 1:
                traceback.print_exc()

    yield emit("\n%d captioned   %d skipped   %d failed" % (done, skipped, failed))


def do_single(file_path, preset, prompt_text, max_new_tokens, max_res,
              use_audio, cpu_offload):
    if not file_path:
        return "Pick a file first."
    path = file_path if isinstance(file_path, str) else file_path.name
    k = C.kind(path)
    if k is None:
        return "Unsupported file type."
    try:
        return C.caption_file(
            path, _prompt_for(k, preset, prompt_text),
            max_new_tokens=max_new_tokens, max_res=max_res,
            use_audio=use_audio, cpu_offload=cpu_offload)
    except Exception as exc:
        traceback.print_exc()
        return "Error: %s" % exc


def on_preset(preset):
    if preset == AUTO:
        return gr.update(value=AUTO_PREVIEW, interactive=False,
                         label="Text  (read-only - Auto picks one of these per file)")
    return gr.update(value=C.PRESETS.get(preset, ""), interactive=True,
                     label="Text  (edit it - this is what the model is told)")


# --------------------------------------------------------------------------- #
# prompt maker
# --------------------------------------------------------------------------- #

REF_LABEL = "Labels the model will see"


def show_refs(files):
    """Which label each dropped file gets, before anything is generated."""
    refs, dropped = C.label_refs(files)
    if not refs and not dropped:
        return gr.update(value="No references yet.", label=REF_LABEL)
    lines = ["%-11s %s" % (label, os.path.basename(path)) for label, path, _ in refs]
    if dropped:
        lines += ["", "not used:"] + ["  " + d for d in dropped]
    return gr.update(value="\n".join(lines), label=REF_LABEL)


def on_builder(name):
    return gr.update(value=C.BUILDERS.get(name, C.SHOT_INSTRUCTION))


def do_build(files, intent, fmt, template, max_new_tokens, max_res, video_fps,
             use_audio, cpu_offload, progress=gr.Progress()):
    def step(i, total, label, what):
        progress((i, total + 1), desc="%s %s" % (label, what) if label else what)

    step(0, 1, "", "Loading the model ...")
    try:
        text, notes, dropped = C.build_prompt(
            files, intent, template, fmt=fmt,
            max_new_tokens=max_new_tokens, max_res=max_res,
            video_fps=video_fps, use_audio=use_audio, cpu_offload=cpu_offload,
            on_step=step)
    except Exception as exc:
        traceback.print_exc()
        return "Error: %s" % exc, gr.update()
    if dropped:
        text += "\n\n---\nnot used:  " + "  |  ".join(dropped)
    seen = "\n\n".join("%s  %s\n%s" % (label, os.path.basename(path), desc)
                       for label, path, _k, desc in notes)
    return text, gr.update(value=seen or "No references.",
                           label="What the model saw in each reference")


# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #

CSS = """
:root, .dark {
  --app-bg:#12141a; --app-panel:#1a1d24; --app-line:#282c35;
  --app-ink:#e6e8ec; --app-dim:#8b9099; --app-accent:#e0a051;
}

/* ---- fill the window, no page scroll ---- */
html, body { height:100%; height:100dvh; margin:0; overflow:hidden; }
body, gradio-app { background:var(--app-bg) !important; }
gradio-app { display:block; height:100vh; height:100dvh; }
.gradio-container { height:100vh !important; height:100dvh !important; max-width:100% !important;
                    width:100% !important; padding:0 !important; }
.app, .fillable, main {
  max-width:none !important; width:100% !important; height:100% !important;
  padding:0 !important; display:flex !important; flex-direction:column !important;
}
main > .wrap, .wrap {
  flex:1 1 auto; min-height:0; display:flex; flex-direction:column;
  padding:0 18px 12px !important; gap:0 !important;
}
/* Gradio wraps top-level children in a column with flex:0 1 auto - that is
   what stops the panels reaching the bottom of the window. */
main.contain, main > .column {
  flex:1 1 auto !important; min-height:0 !important;
  display:flex !important; flex-direction:column !important;
  padding-bottom:0 !important;
}
footer, .built-with, .show-api { display:none !important; }

/* ---- top bar ---- */
#topbar { flex:0 0 auto !important; height:auto !important; min-height:0 !important;
          border-bottom:1px solid var(--app-line);
          padding:12px 4px 10px !important; margin-bottom:12px; }
#topbar > *, #topbar .block, #topbar .html {
  flex:0 0 auto !important; height:auto !important; min-height:0 !important;
  padding:0 !important; }
#topbar h1 { font-size:17px !important; font-weight:600 !important; margin:0 !important;
             color:var(--app-ink) !important; }
#topbar .sub { font-size:12px; color:var(--app-dim); margin-top:3px; }
#topbar .pill { display:inline-block; font-size:10px; letter-spacing:.09em;
                text-transform:uppercase; font-weight:700; padding:3px 8px;
                border-radius:3px; margin-left:9px; vertical-align:2px;
                background:rgba(224,160,81,.14); color:var(--app-accent);
                border:1px solid rgba(224,160,81,.32); }

/* ---- the two panels stretch to the bottom ---- */
#body { flex:1 1 auto !important; min-height:0 !important; gap:14px !important;
        align-items:stretch !important; }
#body > .column { height:100% !important; min-height:0 !important; }
#rail, #stage { background:var(--app-panel); border:1px solid var(--app-line);
                border-radius:8px; padding:13px 14px !important;
                height:100%; min-height:0; }
#rail  { overflow-y:auto; overflow-x:hidden; }
#stage { display:flex !important; flex-direction:column !important; }
#stage .tabs {
  flex:1 1 auto !important; min-height:0 !important;
  display:flex !important; flex-direction:column !important;
}
/* Gradio hides the tabs you are not on with an inline display:none, so this
   must NOT force display - !important here beat the inline rule and left every
   visited tab stacked on the page at once. */
#stage .tabitem {
  flex:1 1 auto !important; min-height:0 !important;
  flex-direction:column !important;
}

/* the log and the single-file caption grow into whatever is left.
   Every wrapper Gradio puts between the tab and the textarea defaults to
   flex:0 1 auto, so each one has to be opened up explicitly. */
#stage .tabitem > .column { flex:1 1 auto !important; min-height:0 !important; }
#logwrap, #capwrap, #pmwrap { flex:1 1 auto !important; min-height:0 !important;
                     display:flex !important; flex-direction:column !important; }
#logwrap .form, #capwrap .form, #pmwrap .form {
                     flex:1 1 auto !important; min-height:0 !important; }
#log, #capbox, #pmbox { flex:1 1 auto !important; min-height:0 !important;
                display:flex !important; flex-direction:column !important; }
#log > label, #capbox > label, #pmbox > label {
                flex:1 1 auto !important; min-height:0 !important;
                display:flex !important; flex-direction:column !important; }
#log .input-container, #capbox .input-container, #pmbox .input-container {
                flex:1 1 auto !important; min-height:0 !important; }
#log textarea, #capbox textarea, #pmbox textarea {
  flex:1 1 auto !important; height:100% !important;
  min-height:0 !important; max-height:none !important; resize:none !important;
}
/* prompt maker: the reference/intent block stays a fixed size at the top */
#pmtop { flex:0 0 auto !important; align-items:stretch !important; }
#pmtop .column { min-height:0 !important; }
#pmrow { flex:0 0 auto !important; }

/* ---- sections and inner blocks ---- */
.sect { font-size:10px; font-weight:700; letter-spacing:.11em; text-transform:uppercase;
        color:var(--app-dim); margin:0 0 8px; padding-bottom:6px;
        border-bottom:1px solid var(--app-line); }
#rail .form, #stage .form, #rail fieldset.block, #stage fieldset.block,
#rail .block, #stage .block {
  background:transparent !important; border:none !important;
  box-shadow:none !important; padding:0 !important;
}
#rail .gap, #stage .gap { gap:9px !important; }

/* ---- typography ---- */
label > span, .gr-box label span { font-size:12px !important; font-weight:600 !important;
        color:var(--app-dim) !important; }
.mono textarea, .mono input {
  font-family:"JetBrains Mono","Cascadia Mono",Consolas,ui-monospace,monospace !important;
  font-size:12.2px !important; line-height:1.65 !important;
  background:#0f1116 !important; color:var(--app-ink) !important;
  border-color:var(--app-line) !important;
}
#log textarea { white-space:pre !important; overflow-x:auto !important; }

/* ---- file-type chips ---- */
[data-testid="checkbox-group"] {
  display:flex !important; flex-direction:row !important; flex-wrap:wrap !important; gap:8px !important;
  align-items:flex-start !important;
}
[data-testid="checkbox-group"] label {
  flex:0 0 auto !important; width:auto !important;
  background:#0f1116 !important; border:1px solid var(--app-line) !important;
  color:var(--app-dim) !important; border-radius:5px !important;
  padding:5px 11px !important; font-size:12.5px !important;
}
[data-testid="checkbox-group"] label.selected {
  background:rgba(224,160,81,.12) !important;
  border-color:rgba(224,160,81,.45) !important; color:var(--app-ink) !important;
}

/* ---- buttons ---- */
button.primary, .primary button { background:var(--app-accent) !important;
        border-color:var(--app-accent) !important; color:#181206 !important;
        font-weight:700 !important; }
.tab-nav button { font-size:13px !important; }
"""


def top_html():
    ok = C.model_present()
    pill = "READY" if ok else "NO MODEL"
    return (
        "<div><h1>ASID Captioner"
        "<span class='pill'>%s</span></h1>"
        "<div class='sub'>%s &nbsp;·&nbsp; 4-bit &nbsp;·&nbsp; "
        "video, gif, image and audio &rarr; <code>.txt</code> beside each file</div></div>"
        % (pill, C.MODEL_NAME)
    )


# --------------------------------------------------------------------------- #

def build():
    with gr.Blocks(title="ASID Captioner", fill_width=True) as demo:

        with gr.Column(elem_id="topbar"):
            gr.HTML(top_html())

        with gr.Row(equal_height=False, elem_id="body"):

            with gr.Column(scale=3, min_width=330, elem_id="rail"):
                gr.HTML("<div class='sect'>Prompt</div>")
                preset = gr.Dropdown(choices=PRESET_NAMES, value=AUTO,
                                     label="Preset", filterable=False)
                prompt_text = gr.Textbox(
                    label="Text  (read-only - Auto picks one of these per file)",
                    lines=6, max_lines=6, value=AUTO_PREVIEW, interactive=False,
                    elem_classes="mono")

                gr.HTML("<div class='sect' style='margin-top:14px'>Settings</div>")
                max_new_tokens = gr.Slider(64, 2048, value=512, step=64,
                                           label="Max caption tokens")
                max_res = gr.Slider(256, 1024, value=512, step=64,
                                    label="Max frame resolution")
                use_audio = gr.Checkbox(True, label="Use the audio track")
                cpu_offload = gr.Checkbox(False, label="CPU offload  (if VRAM runs out)")
                skip_existing = gr.Checkbox(True, label="Skip files that already have a caption")
                caption_ext = gr.Textbox("txt", label="Caption extension")

            with gr.Column(scale=9, elem_id="stage"):
                with gr.Tabs():
                    with gr.Tab("Batch"):
                        folder = gr.Textbox(label="Folder", elem_classes="mono",
                                            placeholder=r"D:\datasets\my_clips")
                        with gr.Row():
                            kinds = gr.CheckboxGroup(choices=list(KIND_LABELS.keys()),
                                                     value=DEFAULT_KINDS,
                                                     label="File types", scale=3)
                            recursive = gr.Checkbox(True, label="Subfolders", scale=1)
                        with gr.Row():
                            scan_btn = gr.Button("Scan", scale=1)
                            run_btn = gr.Button("Caption all", variant="primary", scale=2)
                        with gr.Column(elem_id="logwrap"):
                            out_log = gr.Textbox(label="Progress", lines=8,
                                                 autoscroll=True, buttons=["copy"],
                                                 elem_id="log", elem_classes="mono")

                    with gr.Tab("Single file"):
                        single = gr.File(label="File", type="filepath", height=140)
                        single_btn = gr.Button("Caption", variant="primary")
                        with gr.Column(elem_id="capwrap"):
                            single_out = gr.Textbox(label="Caption", lines=8,
                                                    buttons=["copy"], elem_id="capbox",
                                                    elem_classes="mono")

                    with gr.Tab("Prompt maker"):
                        with gr.Row(equal_height=False, elem_id="pmtop"):
                            with gr.Column(scale=5):
                                refs_in = gr.File(
                                    label="References  -  images, video and audio, mixed",
                                    file_count="multiple", type="filepath", height=132)
                                ref_list = gr.Textbox(
                                    label=REF_LABEL, lines=4,
                                    max_lines=6, value="No references yet.",
                                    interactive=False, elem_classes="mono")
                            with gr.Column(scale=7):
                                intent = gr.Textbox(
                                    label="What you want", lines=4, max_lines=6,
                                    elem_classes="mono",
                                    placeholder="Keep her face from the photo, put her "
                                                "on the rooftop at sunrise, she turns "
                                                "and says one line, warm and calm.")
                                builder = gr.Dropdown(
                                    choices=BUILDER_NAMES, value=BUILDER_NAMES[0],
                                    label="Prompt format", filterable=False)
                                template = gr.Textbox(
                                    label="How to write the shots  (edit freely)",
                                    lines=5, max_lines=7,
                                    value=C.BUILDERS[BUILDER_NAMES[0]],
                                    elem_classes="mono")
                        with gr.Row(elem_id="pmrow"):
                            pm_tokens = gr.Slider(256, 2048, value=600, step=64,
                                                  label="Max prompt tokens")
                            pm_res = gr.Slider(224, 768, value=448, step=32,
                                               label="Max reference resolution")
                            pm_fps = gr.Slider(0.25, 4, value=1.0, step=0.25,
                                               label="Video sample fps")
                        build_btn = gr.Button("Write the prompt", variant="primary")
                        with gr.Column(elem_id="pmwrap"):
                            pm_out = gr.Textbox(label="MiniMax prompt", lines=8,
                                                buttons=["copy"], elem_id="pmbox",
                                                elem_classes="mono")

        preset.change(on_preset, preset, prompt_text)
        scan_btn.click(do_scan, [folder, recursive, kinds], out_log)
        run_btn.click(do_caption,
                      [folder, recursive, kinds, preset, prompt_text, max_new_tokens,
                       max_res, use_audio, cpu_offload, skip_existing, caption_ext],
                      out_log)
        single_btn.click(do_single,
                         [single, preset, prompt_text, max_new_tokens, max_res,
                          use_audio, cpu_offload],
                         single_out)

        refs_in.change(show_refs, refs_in, ref_list)
        builder.change(on_builder, builder, template)
        build_btn.click(do_build,
                        [refs_in, intent, builder, template, pm_tokens, pm_res,
                         pm_fps, use_audio, cpu_offload],
                        [pm_out, ref_list])
    return demo


# force dark mode - the light theme fights the panel colours above
FORCE_DARK = """() => {
  const p = new URLSearchParams(window.location.search);
  if (p.get('__theme') !== 'dark') {
    p.set('__theme', 'dark');
    window.location.search = p.toString();
  }
}"""


def make_theme():
    """Soft, but with the amber label chips flattened back to plain text."""
    return gr.themes.Soft(primary_hue="amber", neutral_hue="slate").set(
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="#8b9099",
        block_label_text_color_dark="#8b9099",
        block_label_border_width="0px",
        block_label_padding="0px 0px 6px 0px",
        block_label_radius="0px",
        block_label_shadow="none",
        block_label_text_size="12px",
        block_label_text_weight="600",
        block_title_background_fill="transparent",
        block_title_text_color="#8b9099",
        block_title_text_color_dark="#8b9099",
        block_title_border_width="0px",
        block_title_padding="0px 0px 6px 0px",
        block_title_radius="0px",
        block_title_text_size="12px",
        block_title_text_weight="600",
    )


if __name__ == "__main__":
    PORT = 7870
    print(banner(PORT), flush=True)
    build().launch(server_name="127.0.0.1", server_port=PORT,
                   inbrowser=True, theme=make_theme(), css=CSS, js=FORCE_DARK,
                   quiet=True)
