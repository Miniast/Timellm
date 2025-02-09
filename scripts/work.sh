model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=12345
num_process=8
batch_size=2
d_model=32
d_ff=128

comment='TimeLLM'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Train.csv \
  --model_id Train \
  --model $model_name \
  --data Train \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
