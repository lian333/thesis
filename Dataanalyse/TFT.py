import copy
from pathlib import Path
import warnings
import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

os.chdir("../../..")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters



# imports for training
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner



# Data loading
train = pd.read_csv(r"C:\Users\lia68085\Thesis\masterarbeit\Dataanalyse\TFT_data\train.csv")

test = pd.read_csv(r"C:\Users\lia68085\Thesis\masterarbeit\Dataanalyse\TFT_data\test.csv")
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

df = pd.concat([train, test], axis = 0, ignore_index=True)
# replace the nan values with 0
#df = df.fillna(0)
# drop the line with num_sold is nan
df = df.dropna(subset=['num_sold'])



# Make sure 'date' is a datetime column
df['date'] = pd.to_datetime(df['date'])

# Correctly define the cutoff date for the training data
training_cutoff = pd.to_datetime("2020-12-31")  # Example date, replace with your actual cutoff

# Convert 'date' to 'time_idx'
df['time_idx'] = (df.date - df.date.min()).dt.days
max_encoder_length = 36
max_prediction_length = 6
# Define the TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    df[df.date <= training_cutoff],
    time_idx="time_idx",
    target="num_sold",
    group_ids=["country", "store", "product"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["country", "store", "product"],
    static_reals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["num_sold"],
)

# Update the validation dataset creation
# Note: You might need to adjust this part based on the actual max `time_idx` and how you've structured it
max_time_idx = df['time_idx'].max()
validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=max_time_idx - max_prediction_length + 1, stop_randomization=True)



# convert datasets to dataloaders for training
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# create PyTorch Lighning Trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",  # run on CPU, if on multiple GPUs, use strategy="ddp"
    gradient_clip_val=0.1,
    limit_train_batches=30,  # 30 batches per epoch
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)

# define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
tft = TemporalFusionTransformer.from_dataset(
    # dataset
    training,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    # loss metric to optimize
    loss=QuantileLoss(),
    # logging frequency
    log_interval=2,
    # optimizer parameters
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find the optimal learning rate
res = Tuner(trainer).lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)
# and plot the result - always visually confirm that the suggested learning rate makes sense
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model on the data - redefine the model with the correct learning rate if necessary
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)