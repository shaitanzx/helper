import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
###import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import launch
###from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

""" -*- coding: UTF-8 -*-
This extension can help you manage your models from civitai.
 It can download preview, add trigger words, open model page and use the prompt from preview image
repo: https://github.com/butaixianran/
"""

###import modules
##from modules import scripts
###from modules import shared
###from modules import script_callbacks
from ch_lib import model
from ch_lib import js_action_civitai
from ch_lib import civitai
from ch_lib import util
from ch_lib import sections
from ch_lib import browser
from ch_lib import scripts

# init
# root path
ROOT_PATH = os.getcwd()

# extension path
EXTENSION_PATH = scripts.basedir()

util.script_dir = EXTENSION_PATH

# default hidden values for civitai helper buttons
BUTTONS = {
    "replace_preview_button": False,
    "open_url_button": False,
    "add_trigger_words_button": False,
    "add_preview_prompt_button": False,
    "rename_model_button": False,
    "remove_model_button": False,
}

model.get_custom_model_folder()


    # ====UI====
shared.gradio_root = gr.Blocks(title="helper").queue()

with shared.gradio_root:
    # with gr.Blocks(css=".block.padded {padding: 10px !important}") as civitai_helper:
        # init
        with gr.Box(elem_classes="ch_box"):
            sections.scan_models_section()

        with gr.Box(elem_classes="ch_box"):
            sections.get_model_info_by_url_section()

        with gr.Box(elem_classes="ch_box"):
            gr.Markdown("### Download Model")
            with gr.Tab("Single", elem_id="ch_dl_single_tab"):
                sections.download_section()
            with gr.Tab("Batch Download"):
                sections.download_multiple_section()

        with gr.Box(elem_classes="ch_box"):
            sections.scan_for_duplicates_section()








shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=True,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
