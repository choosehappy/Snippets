# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from moviepy.editor import *
import os
import glob

for fname in glob.glob('*.avi'):
    clip = (VideoFileClip(fname)).resize(0.25).speedx(2)
    clip.write_gif(fname.replace("avi","gif"))
# -


