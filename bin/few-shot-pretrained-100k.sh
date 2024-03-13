
allow_skip_exp=True
eval_before_training=True
balanced_ibc=True

train_batch_size=2
grad_accum_factor=1

lr=0.002
re='^[0-9]+$'

cuda_device=0

num_steps=0
eval_epoch_interval=0

for model in 't03b'
do
  for num_shot in 4
  do
    for dataset in seer
    do
      eval_before_training=False
      num_steps=$(( 30 * ($num_shot / $train_batch_size)))
      eval_epoch_interval=30

      if ! [[ $num_shot =~ $re ]]; then
        if [[ $dataset = *"income"* ]]; then
          num_steps=295000
        fi
        if [[ $dataset = *"car"* ]]; then
          num_steps=10500
        fi
        if [[ $dataset = *"heart"* ]]; then
          num_steps=5600
        fi
        if [[ $dataset = *"diabetes"* ]]; then
          num_steps=4700
        fi
        if [[ $dataset = *"bank"* ]]; then
          num_steps=272000
        fi
        if [[ $dataset = *"blood"* ]]; then
          num_steps=4520
        fi
        if [[ $dataset = *"calhousing"* ]]; then
          num_steps=124000
        fi
        if [[ $dataset = *"creditg"* ]]; then
          num_steps=6000
        fi
        if [[ $dataset = *"jungle"* ]]; then
          num_steps=270000
        fi
        if [[ $dataset = *"seer"* ]]; then
          num_steps=4700
        fi
      fi


      for seed in 42
      do
        CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=t-few/configs HF_HOME=content/.cache/huggingface \
        python3.8 -m t-few.src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="t-few/pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
        exp_name=${model}_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
        batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}

        echo "Logging GPU memory usage..."
        python3.8 -c "import torch; print(f'Current memory allocated: {torch.cuda.memory_allocated() / 1e9} GB'); print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB')"
      done

      
    done
  
  done
done
