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

from md_lib import md_config
from md_lib import model
from md_lib import js_action_civitai
from md_lib import civitai
from md_lib import util
from md_lib import sections

txt2img_prompt = 'txt2img_prompt'
txt2img_neg_prompt = 'txt2img_neg_prompt'
img2img_prompt = 'img2img_prompt'
img2img_neg_prompt = 'img2img_neg_prompt'
# Used by some elements to pass messages to python
js_msg_txtbox = gr.Textbox(
        label="Request Msg From Js",
        visible=False,
        lines=1,
        value="",
        elem_id="ch_js_msg_txtbox"
    )

    # ====UI====
shared.gradio_root = gr.Blocks(title="helper").queue()
def apply_settings(api_key,example):
        md_config.ch_civiai_api_key=api_key
        md_config.ch_download_examples=example
        new_setting="Download Example Images Locally = "+ str(md_config.ch_download_examples)+", Civitai API key="+str(md_config.ch_civiai_api_key)
        return setting_log.update(value=new_setting)   
with shared.gradio_root:
        with gr.Box(elem_classes="ch_box"):
            with gr.Column(elem_classes="justify-bottom"):
                    gr.Markdown("### Settings")
                    dl_civital_model_example_image_local=gr.Checkbox(
                      label="Download Example Images Locally",
                      value=md_config.ch_download_examples,
                      interactive=True)
                    dl_civitai_apikey = gr.Textbox(
                        label="Civitai API key",
                        lines=1,
                        max_lines=1,
                        value=md_config.ch_civiai_api_key,
                        placeholder="Civitai API key",
                        elem_id="ch_dl_url"
                    )
                    dl_apply = gr.Button(
                      value="Apply setting",
                      elem_classes="ch_vmargin",
                      variant="primary",
                      elem_id="ch_download_model_button"
                    )
                    setting_log = gr.Markdown(value="Download Example Images Locally = "+ str(md_config.ch_download_examples)+", Civitai API key="+str(md_config.ch_civiai_api_key))
                    dl_apply.click(apply_settings,inputs=[dl_civitai_apikey,dl_civital_model_example_image_local],outputs=setting_log)


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

        with gr.Box(elem_classes="ch_box"):
            sections.check_new_versions_section(js_msg_txtbox)

        # ====Footer====
        gr.HTML(f"<center>{util.SHORT_NAME} version: {util.VERSION}</center>")

        # ====hidden component for js, not in any tab====
        js_msg_txtbox.render()
        py_msg_txtbox = gr.Textbox(
            label="Response Msg From Python",
            visible=False,
            lines=1,
            value="",
            elem_id="ch_py_msg_txtbox"
        )

        js_open_url_btn = gr.Button(
            value="Open Model Url",
            visible=False,
            elem_id="ch_js_open_url_btn"
        )
        js_add_trigger_words_btn = gr.Button(
            value="Add Trigger Words",
            visible=False,
            elem_id="ch_js_add_trigger_words_btn"
        )
        js_use_preview_prompt_btn = gr.Button(
            value="Use Prompt from Preview Image",
            visible=False,
            elem_id="ch_js_use_preview_prompt_btn"
        )
        js_rename_card_btn = gr.Button(
            value="Rename Card",
            visible=False,
            elem_id="ch_js_rename_card_btn"
        )
        js_remove_card_btn = gr.Button(
            value="Remove Card",
            visible=False,
            elem_id="ch_js_remove_card_btn"
        )
"""
        # ====events====
        # js action
        js_open_url_btn.click(
            js_action_civitai.open_model_url,
            inputs=[js_msg_txtbox],
            outputs=py_msg_txtbox
        )


        js_add_trigger_words_btn.click(
            js_action_civitai.add_trigger_words,
            inputs=[js_msg_txtbox],
            outputs=[
                txt2img_prompt, img2img_prompt
            ]
        )
        js_use_preview_prompt_btn.click(
            js_action_civitai.use_preview_image_prompt,
            inputs=[js_msg_txtbox],
            outputs=[
                txt2img_prompt, txt2img_neg_prompt,
                img2img_prompt, img2img_neg_prompt
            ]
        )
        js_rename_card_btn.click(
            js_action_civitai.rename_model_by_path,
            inputs=[js_msg_txtbox],
            outputs=py_msg_txtbox
        )
        js_remove_card_btn.click(
            js_action_civitai.remove_model_by_path,
            inputs=[js_msg_txtbox],
            outputs=py_msg_txtbox
        )
"""
shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=True,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
