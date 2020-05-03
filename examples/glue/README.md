# GLUE Benchmark
### setup
1. pip install -r requirements.txt
2. get data: `python3 ../../utils/download_glue_data.py`
3. `mkdir mrpc-pl-bert`
4. `export PYTHONPATH="../":"../../":"${PYTHONPATH}"`
### train and evaluate
`CUDA_VISIBLE_DEVICES=1 python3 run_pl_glue.py --data_dir ./glue_data/MRPC/ --task mrpc --model_name_or_path bert-base-cased --output_dir mrpc-pl-bert --max_seq_length  128 --learning_rate 2e-5 --num_train_epochs 3 --train_batch_size 8 --seed 2 --n_gpu 1 --do_train --do_predict`

after 3 epochs and ~5 minutes: 
```python
# TEST RESULTS
{'acc': 0.8676470588235294,
 'acc_and_f1': 0.8877484440875327,
 'avg_test_loss': 0.6455589532852173,
 'f1': 0.9078498293515359,
 'val_loss': 0.6455589532852173}
```