Train GCNN model

  $ cp -R ${TESTDIR}/* .

# Test mse_loss model

  $ train_gcnn.py \
  > --data_path files/data.lz4 \
  > --record_path ${CRAMTMP}/temp.csv \
  > --model_path ${CRAMTMP}/temp.h5 \
  > --metrics_path ${CRAMTMP}/temp.dat \
  > --epochs 2 \
  > --batch_size 2 \
  > --learning_rate 0.1 \
  > --channels 2 2 \
  > --n_layers 2 \
  > --seed 0 \
  > "mse_loss"
  $ diff files/mse_train.csv ${CRAMTMP}/temp.csv
  $ diff files/mse_metrics.dat ${CRAMTMP}/temp.dat
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/mse_model.h5 /model_weights
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/mse_model.h5 /optimizer_weights

# Test maxlike_mse_loss model

  $ train_gcnn.py \
  > --data_path files/data.lz4 \
  > --record_path ${CRAMTMP}/temp.csv \
  > --model_path ${CRAMTMP}/temp.h5 \
  > --metrics_path ${CRAMTMP}/temp.dat \
  > --epochs 2 \
  > --batch_size 2 \
  > --learning_rate 0.1 \
  > --channels 2 2 \
  > --n_layers 2 \
  > --seed 0 \
  > "maxlike_mse_loss" 
  $ diff files/max_train.csv ${CRAMTMP}/temp.csv
  $ diff files/max_metrics.dat ${CRAMTMP}/temp.dat
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/max_model.h5 /model_weights
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/max_model.h5 /optimizer_weights

# Test maxlike_cse_loss model

  $ train_gcnn.py \
  > --data_path files/data.lz4 \
  > --record_path ${CRAMTMP}/temp.csv \
  > --model_path ${CRAMTMP}/temp.h5 \
  > --metrics_path ${CRAMTMP}/temp.dat \
  > --epochs 2 \
  > --batch_size 2 \
  > --learning_rate 0.1 \
  > --channels 2 2 \
  > --n_layers 2 \
  > --seed 0 \
  > "maxlike_cse_loss"
  $ diff files/cse_train.csv ${CRAMTMP}/temp.csv
  $ diff files/cse_metrics.dat ${CRAMTMP}/temp.dat
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/cse_model.h5 /model_weights
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/cse_model.h5 /optimizer_weights

# Test maxlike_tobit_loss model

  $ train_gcnn.py \
  > --data_path files/data.lz4 \
  > --record_path ${CRAMTMP}/temp.csv \
  > --model_path ${CRAMTMP}/temp.h5 \
  > --metrics_path ${CRAMTMP}/temp.dat \
  > --epochs 2 \
  > --batch_size 2 \
  > --learning_rate 0.1 \
  > --channels 2 2 \
  > --n_layers 2 \
  > --seed 0 \
  > "maxlike_tobit_loss"
  $ diff files/tobit_train.csv ${CRAMTMP}/temp.csv
  $ diff files/tobit_metrics.dat ${CRAMTMP}/temp.dat
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/tobit_model.h5 /model_weights
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/tobit_model.h5 /optimizer_weights
