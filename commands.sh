#resnet18
python main.py \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep' --data-path='.'


#ela-big
python main.py \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep' \
    --data-path='.'\
    --attention-type "ELA" \

#ela tiny
python main.py \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep' \
    --data-path='.' \
    --attention-type "ELA" \
    --ela-numgroup 32 \
    --ela-kernelsize 5 \ 

#ela large
python main.py \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep' \
    --data-path='.' \
    --attention-type "ELA" \
    --ela-numgroup 16 \
    --ela-kernelsize 7 \
    --ela-group-setting "channel/8"  

#ela small
python main.py \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep'\
    --data-path='.'\
    --attention-type "ELA" \
    --ela-numgroup 16 \
    --ela-kernelsize 5 \
    --ela-group-setting "channel/8"