import argparse
import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import normalized_mutual_info_score

# config.py に model_path などを渡すため
from data import *

from utils.prototype_split import analyze_true_category_mixing


parser_local = argparse.ArgumentParser()