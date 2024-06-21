root_path="."
python3 main.py --tag pm621  --root_path $root_path \
    --lr 2e-5 --warmup_steps 120 --do_train --use_bart
    