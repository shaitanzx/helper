import os
import gradio as gr
####import modules
import scripts
###from modules import shared
###from modules import script_callbacks
import model
import js_action_civitai
import civitai
import util
import sections
import browser


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

with gr.Blocks(analytics_enabled=False) as dm2:
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

dm2.launch()
