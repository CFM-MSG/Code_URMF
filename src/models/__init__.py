#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from models.bert import BertClf

from models.image import ImageClf

from models.late_fusion import MultimodalLateFusionClf

MODELS = {
    "bert": BertClf,
    "img": ImageClf,
    'latefusion':MultimodalLateFusionClf,
}


def get_model(args):
    # print(args.model)
    return MODELS[args.model](args)
