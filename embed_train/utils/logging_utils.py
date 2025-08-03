import os
import logging
from embed_train.config import OUTPUT_ROOT
def setup_logging():
    logging.root.handlers = []
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="train_saint.log",
        filemode="a"
    )