Train GCNN model

  $ cp -R ${TESTDIR}/* .

# Test training record

  $ train_gcnn.py \
  > --data_path files/data.gz \
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
  $ diff files/train.csv ${CRAMTMP}/temp.csv
  $ diff files/metrics.dat ${CRAMTMP}/temp.dat

# Test output model

  $ train_gcnn.py \
  > --data_path files/data.gz \
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
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/model.h5 /model_weights
  $ h5diff -d 1e-1 ${CRAMTMP}/temp.h5 files/model.h5 /optimizer_weights