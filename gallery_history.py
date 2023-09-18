"""
How to use:
1. Create a Space with a Persistent Storage attached. Filesystem will be available under `/data`.
2. Add `hf_oauth: true` to the Space metadata (README.md). Make sure to have Gradio>=3.41.0 configured.
3. Add `HISTORY_FOLDER` as a Space variable (example. `"/data/history"`).
4. Add `filelock` as dependency in `requirements.txt`.
5. Add history gallery to your Gradio app:
    a. Add imports: `from gallery_history import fetch_gallery_history, show_gallery_history`
    a. Add `history = show_gallery_history()` within `gr.Blocks` context.
    b. Add `.then(fn=fetch_gallery_history, inputs=[prompt, result], outputs=history)` on the generate event.
"""
import json
import os
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr
from filelock import FileLock

_folder = os.environ.get("HISTORY_FOLDER")
if _folder is None:
    print(
        "'HISTORY_FOLDER' environment variable not set. User history will be saved "
        "locally and will be lost when the Space instance is restarted."
    )
    _folder = Path(__file__).parent / "history"
HISTORY_FOLDER_PATH = Path(_folder)

IMAGES_FOLDER_PATH = HISTORY_FOLDER_PATH / "images"
IMAGES_FOLDER_PATH.mkdir(parents=True, exist_ok=True)


def show_gallery_history():
    gr.Markdown(
        "## Your past generations\n\n(Log in to keep a gallery of your previous generations."
        " Your history will be saved and available on your next visit.)"
    )
    with gr.Column():
        with gr.Row():
            gr.LoginButton(min_width=250)
            gr.LogoutButton(min_width=250)
        gallery = gr.Gallery(
            label="Past images",
            show_label=True,
            elem_id="gallery",
            object_fit="contain",
            columns=4,
            height=512,
            preview=False,
            show_share_button=False,
            show_download_button=False,
        )
        gr.Markdown(
            "Make sure to save your images from time to time, this gallery may be deleted in the future."
        )
        gallery.attach_load_event(fetch_gallery_history, every=None)
        return gallery


def fetch_gallery_history(
    prompt: Optional[str] = None,
    result: Optional[np.ndarray] = None,
    user: Optional[gr.OAuthProfile] = None,
):
    if user is None:
        return []
    try:
        if prompt is not None and result is not None:  # None values means no new images
            new_image = Image.fromarray(result, 'RGB')
            return _update_user_history(user["preferred_username"], new_image, prompt)
        else:
            return _read_user_history(user["preferred_username"])
    except Exception as e:
        raise gr.Error(f"Error while fetching history: {e}") from e


####################
# Internal helpers #
####################


def _read_user_history(username: str) -> List[Tuple[str, str]]:
    """Return saved history for that user."""
    with _user_lock(username):
        path = _user_history_path(username)
        if path.exists():
            return json.loads(path.read_text())
        return []  # No history yet


def _update_user_history(
    username: str, new_image: Image.Image, prompt: str
) -> List[Tuple[str, str]]:
    """Update history for that user and return it."""
    with _user_lock(username):
        # Read existing
        path = _user_history_path(username)
        if path.exists():
            images = json.loads(path.read_text())
        else:
            images = []  # No history yet

        # Copy image to persistent folder
        images = [(_copy_image(new_image), prompt)] + images

        # Save and return
        path.write_text(json.dumps(images))
        return images


def _user_history_path(username: str) -> Path:
    return HISTORY_FOLDER_PATH / f"{username}.json"


def _user_lock(username: str) -> FileLock:
    """Ensure history is not corrupted if concurrent calls."""
    return FileLock(f"{_user_history_path(username)}.lock")


def _copy_image(new_image: Image.Image) -> str:
    """Copy image to the persistent storage."""
    dst = str(IMAGES_FOLDER_PATH / f"{uuid4().hex}.png")
    new_image.save(dst)
    return dst