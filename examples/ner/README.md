## Named Entity Recognition

### setup on HPC
0. `pip install -r requirements.txt`
1. `OMP_NUM_THREADS=8 bash download_data.sh`
2. `python preprocess.py --model_name_or_path bert-base-multilingual-cased --max_seq_length 128`
3. `export PYTHONPATH="../":"${PYTHONPATH}"`
4. to download pretrained model: `OMP_NUM_THREADS=8 python3 run_pl_ner.py --data_dir ./ --labels ./labels.txt --model_name_or_path $BERT_MODEL --do_train`
### train & evaluate

```shell script
python3 run_pl_ner.py --data_dir ./ \
--labels ./labels.txt \
--model_name_or_path bert-base-multilingual-cased  \
--output_dir checkpoints \
--max_seq_length  128 \
--num_train_epochs 3 \
--train_batch_size 32 \
--seed 1 \
--do_train \
--do_predict
```
after 3 epochs in ~20 minutes: 
```shell script
TEST RESULTS
{'avg_test_loss': tensor(0.0733),
 'f1': 0.8625160051216388,
 'precision': 0.8529597974042419,
 'recall': 0.8722887665911299,
 'val_loss': tensor(0.0733)}

```