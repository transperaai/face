import os
import re
import tempfile
from urllib.parse import quote
import numpy as np
import gradio as gr
from PIL import Image

import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path

DEFAULT_OUTPUT_DIR = "/workspace/outputs"
NAME_PREFIX = "a"


def _next_index(output_dir, prefix=NAME_PREFIX, ext=".jpg"):
    """Scan output_dir for existing a<N>.jpg files and return the next number in sequence.

    Scanning the folder (rather than an in-memory counter) means numbering survives
    app restarts and keeps going across separate conversion runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(ext)}$')
    max_n = 0
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _setup_globals(source_path, target_path, output_path, do_enhancer):
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = normalize_output_path(source_path, target_path, output_path)
    roop.globals.frame_processors = ["face_swapper", "face_enhancer"] if do_enhancer else ["face_swapper"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()


def swap_face_single(source_file, target_file, do_enhancer, output_dir):
    """Swap one source face into one target image and save it as the next aN.jpg."""
    if source_file is None or target_file is None:
        return None, "Upload a source and target image first."

    output_dir = (output_dir or DEFAULT_OUTPUT_DIR).strip() or DEFAULT_OUTPUT_DIR

    work_dir = tempfile.mkdtemp(prefix="faceswap_")
    source_path = os.path.join(work_dir, "source.jpg")
    target_path = os.path.join(work_dir, "target.jpg")

    Image.fromarray(source_file).save(source_path)
    Image.fromarray(target_file).save(target_path)

    index = _next_index(output_dir)
    output_path = os.path.join(output_dir, f"{NAME_PREFIX}{index}.jpg")

    _setup_globals(source_path, target_path, output_path, do_enhancer)

    for fp in get_frame_processors_modules(roop.globals.frame_processors):
        if not fp.pre_check():
            return None, "A frame processor failed its pre-check."

    start()
    if os.path.exists(roop.globals.output_path):
        return roop.globals.output_path, f"Saved to {roop.globals.output_path}"
    return None, "Swap failed."


def swap_face_bulk(source_file, target_files, do_enhancer, output_dir, progress=gr.Progress()):
    """Swap source face into every target image, saving each as the next aN.jpg in sequence."""
    if source_file is None or not target_files:
        return [], "Upload a source and at least one target image first."

    output_dir = (output_dir or DEFAULT_OUTPUT_DIR).strip() or DEFAULT_OUTPUT_DIR

    work_dir = tempfile.mkdtemp(prefix="faceswap_")
    source_path = os.path.join(work_dir, "source.jpg")
    Image.fromarray(source_file).save(source_path)

    start_index = _next_index(output_dir)
    saved_paths = []
    total = len(target_files)

    for i, target_arr in enumerate(target_files):
        progress(i / total, desc=f"Processing {i+1}/{total}")
        target_path = os.path.join(work_dir, f"target_{i:04d}.jpg")
        output_path = os.path.join(output_dir, f"{NAME_PREFIX}{start_index + i}.jpg")

        Image.fromarray(target_arr).save(target_path)
        _setup_globals(source_path, target_path, output_path, do_enhancer)

        ok = True
        for fp in get_frame_processors_modules(roop.globals.frame_processors):
            if not fp.pre_check():
                ok = False
                break

        if ok:
            start()
            if os.path.exists(roop.globals.output_path):
                saved_paths.append(roop.globals.output_path)

    progress(1.0, desc="Done")

    if saved_paths:
        first = os.path.basename(saved_paths[0])
        last = os.path.basename(saved_paths[-1])
        status = f"Saved {len(saved_paths)} image(s) to {output_dir} ({first} – {last})"
    else:
        status = "No images were saved."

    return saved_paths, status


def _to_download_items(paths):
    """Build {url, filename} pairs the browser can use to download each file individually."""
    return [
        {"url": "/file=" + quote(p, safe="/"), "filename": os.path.basename(p)}
        for p in paths
    ]


# Triggers one browser download per item, staggered slightly so browsers don't
# silently block a burst of simultaneous downloads. No zip involved - each file
# lands in the browser's normal download location as its own image.
DOWNLOAD_ALL_JS = """
(items) => {
    items.forEach((item, i) => {
        setTimeout(() => {
            const a = document.createElement('a');
            a.href = item.url;
            a.download = item.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }, i * 400);
    });
}
"""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Face Swap") as app:
    gr.Markdown("# Face Swap")
    gr.Markdown("Upload a **source** face and one or more **target** images, then click Run.")

    with gr.Tab("Single"):
        with gr.Row():
            src_single = gr.Image(label="Source (face to use)")
            tgt_single = gr.Image(label="Target")
        enh_single = gr.Checkbox(label="Face Enhancer", value=False)
        out_dir_single = gr.Textbox(label="Output folder (on the server)", value=DEFAULT_OUTPUT_DIR)
        run_single = gr.Button("Swap", variant="primary")
        out_single = gr.Image(label="Result")
        status_single = gr.Textbox(label="Status", interactive=False)

        run_single.click(
            fn=swap_face_single,
            inputs=[src_single, tgt_single, enh_single, out_dir_single],
            outputs=[out_single, status_single],
        )

    with gr.Tab("Bulk Hopper"):
        gr.Markdown("Upload a single **source** face and as many **target** images as you like.")
        with gr.Row():
            src_bulk = gr.Image(label="Source (face to use)")
        tgt_bulk = gr.File(
            label="Target Images (select multiple)",
            file_count="multiple",
            file_types=["image"],
        )
        enh_bulk = gr.Checkbox(label="Face Enhancer", value=False)
        out_dir_bulk = gr.Textbox(label="Output folder (on the server)", value=DEFAULT_OUTPUT_DIR)
        run_bulk = gr.Button("Swap All", variant="primary")

        gallery = gr.Gallery(label="Results", columns=4, height="auto")
        status_bulk = gr.Textbox(label="Status", interactive=False)
        download_all_btn = gr.Button("Download All (separate images, no zip)")
        download_items_state = gr.JSON(visible=False)

        def _bulk_wrapper(src, files, enh, output_dir, progress=gr.Progress()):
            if src is None or not files:
                return [], "Upload a source and at least one target image first.", []
            # files from gr.File are file paths when file_count="multiple"
            target_arrays = []
            for f in files:
                path = f.name if hasattr(f, "name") else f
                target_arrays.append(np.array(Image.open(path).convert("RGB")))
            imgs, status = swap_face_bulk(src, target_arrays, enh, output_dir, progress)
            return imgs, status, _to_download_items(imgs)

        run_bulk.click(
            fn=_bulk_wrapper,
            inputs=[src_bulk, tgt_bulk, enh_bulk, out_dir_bulk],
            outputs=[gallery, status_bulk, download_items_state],
        )

        download_all_btn.click(
            fn=None,
            inputs=[download_items_state],
            outputs=[],
            js=DOWNLOAD_ALL_JS,
        )


if __name__ == "__main__":
    # allowed_paths lets Gradio serve files from the output folder for browser
    # download - if you change the output folder, keep it under /workspace.
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=["/workspace"])
