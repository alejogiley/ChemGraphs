Prepare Graph Dataset

  $ cp -R ${TESTDIR}/* .

# Test GraphDB dataset

  $ setup_dataset.py --binding files/data.sdf --data_path ${CRAMTMP} --file_name "temp" --metric_type "Ki"
  $ zdiff files/data.gz ${CRAMTMP}/temp.gz

# Test BindingDB dataset

  $ setup_dataset.py --binding files/data.sdf --data_path ${CRAMTMP} --file_name "temp" --metric_type "Ki"
  $ diff files/data.npz ${CRAMTMP}/temp.npz