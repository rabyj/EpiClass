# epi_ml
Use machine learning on epigenomic data

## Known problems

- **training_prediction** method of the **Analysis** class fails because of a dimension problem. It seems to be linked to the training set oversampling. This problem might also apply to training confusion matrix creation.
- Prediction on future data (e.g. new test set) can only be made on data sets that contain the exact same labels set as the training set. If labels are missing, use of one signal per missing label from the training data is recommended.
