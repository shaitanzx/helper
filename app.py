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

from civitai_help import model
from civitai_help import js_action_civitai
from civitai_help import civitai
from civitai_help import util
from civitai_help import sections
from civitai_help import browser
from civitai_help import scripts


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
js_msg_txtbox = gr.Textbox(
        label="Request Msg From Js",
        visible=False,
        lines=1,
        value="",
        elem_id="ch_js_msg_txtbox"
    )
###nsfw_civitai="XXX"
def change_nsfw(nsfw_drop):
    global nsfw_civitai
    nsfw_civitai=nsfw_drop
    return nsfw_civitai

shared.gradio_root = gr.Blocks(title="helper").queue()

with shared.gradio_root:
        # init
        with gr.Row(elem_classes="ch_box"):
            nsfw_drop=gr.Dropdown(choices=list(civitai.NSFW_LEVELS.keys()),label="NSFW_LEVELS",value=list(civitai.NSFW_LEVELS.keys())[0],multiselect=False,interactive=True,elem_classes="ch_vpadding")
            change_nsfw(list(civitai.NSFW_LEVELS.keys())[0])
    
        with gr.Row(elem_classes="ch_box"):
            sections.scan_models_section()

        with gr.Row(elem_classes="ch_box"):
            sections.get_model_info_by_url_section()

        with gr.Row(elem_classes="ch_box"):
            gr.Markdown("### Download Model")
            with gr.Tab("Single", elem_id="ch_dl_single_tab"):
                sections.download_section()
            with gr.Tab("Batch Download"):
                sections.download_multiple_section()

        with gr.Row(elem_classes="ch_box"):
            sections.scan_for_duplicates_section()

        with gr.Row(elem_classes="ch_box"):
            sections.check_new_versions_section(js_msg_txtbox)

        nsfw_drop.change(change_nsfw, inputs=nsfw_drop, outputs=None)

# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=True,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
