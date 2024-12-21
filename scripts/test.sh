model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=12345
num_process=1
batch_size=1
d_model=32
d_ff=128

comment='TimeLLM'

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 12345 run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ \
  --model_id Train \
  --model TimeLLM \
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
  --d_model 32 \
  --d_ff 128 \
  --batch_size 12 \
  --llm_layers 32 \
  --model_comment TimeLLM


