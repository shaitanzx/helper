import os
import io
import json
##import numpy as np
import cv2

import gradio as gr

#import modules.scripts as scripts
#from modules import script_callbacks
#from modules.shared import opts
#from modules.paths import models_path

#from basicsr.utils.download_util import load_file_from_url




body_estimation = None
presets_file = os.path.join(scripts.basedir(), "presets.json")
presets = {}

try: 
  with open(presets_file) as file:
    presets = json.load(file)
except FileNotFoundError:
  pass

def pil2cv(in_image):
  out_image = np.array(in_image, dtype=np.uint8)

  if out_image.shape[2] == 3:
      out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
  return out_image

def candidate2li(li):
  res = []
  for x, y, *_ in li:
    res.append([x, y])
  return res

def subset2li(li):
  res = []
  for r in li:
    for c in r:
      res.append(c)
  return res

#class Script(scripts.Script):
#  def __init__(self) -> None:
#    super().__init__()
#
#  def title(self):
#    return "OpenPose Editor"
#
#  def show(self, is_img2img):
#    return scripts.AlwaysVisible
#
#  def ui(self, is_img2img):
#    return ()