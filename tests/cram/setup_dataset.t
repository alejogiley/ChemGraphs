Prepare Graph Dataset

  $ cp -R ${TESTDIR}/* .

# Test GraphDB dataset

  $ setup_dataset.py --binding files/data.sdf --data_path ${CRAMTMP} --file_name "temp" --metric_type "Ki" > /dev/null 2>&1
  $ python scripts/compare_data.py files/data.lz4 ${CRAMTMP}/temp.lz4
  Success! Datasets have no differences

# Test wrong number of graphs

  $ setup_dataset.py --binding files/data.sdf --data_path ${CRAMTMP} --file_name "temp" --metric_type "IC50"  > /dev/null 2>&1
  $ python scripts/compare_data.py files/data.lz4 ${CRAMTMP}/temp.lz4 2>err
  Error! Datasets do not have same number of graphs
  [1]
