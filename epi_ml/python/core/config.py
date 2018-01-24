import os.path

CORE_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_PATH = os.path.join(CORE_DIR, "tb")
DATA_PATH = os.path.join(CORE_DIR, "data")
CHROM_PATH = os.path.join(DATA_PATH, "saccer3.can.chrom.sizes")
META_PATH = os.path.join(DATA_PATH, "sacCer3_GEO_2016-07.json")
