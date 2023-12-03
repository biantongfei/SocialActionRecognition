import os
import json
import random

import numpy
import numpy as np
from torch.utils.data import Dataset

sigma = '100501'
testset_rate = 0.1
coco_point_num = 133
halpe_point_num = 136