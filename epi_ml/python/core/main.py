import os
import sys

import data
import model

def main(args):
    model.Model(data.EpiData("publishing_group"))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(sys.argv[1:])
