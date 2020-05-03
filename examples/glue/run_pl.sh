export OUTPUT_DIR=${PWD}/mrpc-pl-bert
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../":"../../":"${PYTHONPATH}"

python3 run_pl_glue.py --data_dir ./glue_data/MRPC/ \
--task mrpc \
--model_name_or_path bert-base-cased \
--output_dir $OUTPUT_DIR \
--max_seq_length  128 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--train_batch_size 8 \
--seed 2 \
--do_train \
--do_predict
