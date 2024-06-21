root_path="."
python3 main.py --tag pm621 --do_train \
    --data_path origin_data \
    --root_path $root_path --lr 2e-5 --warmup_steps 120 
    

