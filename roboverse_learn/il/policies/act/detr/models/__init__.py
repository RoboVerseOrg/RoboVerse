# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) 2023 Tony Z. Zhao
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Adapted from ACT's fork of DETR: DETR's models/__init__.py (Apache-2.0, Facebook) rewired by
# ACT (https://github.com/tonyzhaozh/act, MIT, Tony Z. Zhao) to build its CVAE — build_ACT_model
# and build_CNNMLP_model have no counterpart in DETR (https://github.com/facebookresearch/detr).
# Changes: none (vendored verbatim; trailing newline added).
# Full license: roboverse_learn/il/policies/act/detr/LICENSE (Apache-2.0), roboverse_learn/il/policies/act/LICENSE (MIT)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)
