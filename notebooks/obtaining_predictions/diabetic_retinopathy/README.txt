We used the original model made publicly available at https://github.com/JeffreyDF/kaggle_diabetic_retinopathy (won 5th place at the Kaggle Diabetic Retinopathy detection challenge)

Our code for making the predictions using one eye at a time is at https://github.com/kundajelab/kaggle_diabetic_retinopathy/blob/26ca72b09393d9c4360d635f7caa90aaf4d6744a/notebooks/OneEyeAtATimeValidationSetPredictions.ipynb

For each eye, the predictions were made over different rotations and flips in order to get the final predictions, which were prepared for upload to Zenodo with this script: https://github.com/kundajelab/kaggle_diabetic_retinopathy/blob/26ca72b09393d9c4360d635f7caa90aaf4d6744a/notebooks/prepare_data_for_zenodo_upload/prepare_data.py
