import os
import sys
import warnings
warnings.simplefilter("ignore")
import data
import model
import trainer
import config
import os.path

def main(args):
    #my_data = data.EpiData("assay")
    my_data = data.EpiData("publishing_group")
    input_size = my_data.train.signals[0].size
    ouput_size = my_data.train.labels[0].size
    my_model = model.Dense(input_size, ouput_size)
    my_trainer = trainer.Trainer(my_data, my_model)
    my_trainer.train()
    my_trainer.metrics()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
