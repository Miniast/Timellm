model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=32

master_port=12345
num_process=8
batch_size=8
d_model=32
d_ff=128

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_stock_predict.py \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \