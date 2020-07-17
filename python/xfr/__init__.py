# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import os

# Create variables to common directories
xfr_root = os.path.dirname(os.path.dirname(__path__[0]))
inpaintgame2_dir = os.path.join(xfr_root, 'data/inpainting-game/IJBC')
output_dir = os.path.join(xfr_root, 'output')

# Saliency map results of inpainting game from individual networks.
inpaintgame_saliencymaps_dir = os.path.join(
    xfr_root, 'data',
    'inpainting-game-saliency-maps')
