# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Laboratory asset configurations."""

from .beakers import *
from .bottles import BOTTLE_CFG, BOTTLE_INST_CFG
from .flasks import FLASK_CFG, FLASK_INST_CFG
from .trays import TRAY_CFG, TRAY_INST_CFG

__all__ = [
    "BOTTLE_CFG",
    "BOTTLE_INST_CFG",
    "FLASK_CFG",
    "FLASK_INST_CFG",
    "TRAY_CFG",
    "TRAY_INST_CFG",
]
