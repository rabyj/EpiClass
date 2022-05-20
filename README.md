# epi_ml
Use machine learning on epigenomic data

See `input-format` folder for examples of mandatory files.

~~~text
usage: main.py [-h] [--offline] [--predict] [--model MODEL]
               category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metatada category to analyse.
  hyperparameters  A json file containing model hyperparameters.
  hdf5             A file with hdf5 filenames. Use absolute path!
  chromsize        A file with chrom sizes.
  metadata         A metadata JSON file.
  logdir           Directory for the output logs.

optional arguments:
  -h, --help       show this help message and exit
  --offline        Will log data offline instead of online. Currently cannot merge comet-ml offline
                   outputs.
  --predict        Enter prediction mode. Will use all data for the test set. Overwrites hparameter file
                   setting. Default mode is training mode.
  --model MODEL    Directory from which to load the desired model. Default is logdir.
~~~
