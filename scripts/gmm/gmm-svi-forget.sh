cd $1

python main.py \
    --exp-name gmm \
    --exp-type  forget \
    --is-vi \
    --gmm-kk 4 \
    --gmm-std 1 \
    --dataset gmm2d \
    --batch-size 64 \
    --burn-in-steps 2000 \
    --eval-freq 100 \
    --lr 2 \
    --ifs-scaling 1 \
    --ifs-iter-T 32 \
    --ifs-iter-bs 64 \
    --ifs-rm-bs 4 \
    --ifs-kill-num 800 \
    --save-dir ./exp_data/gmm/svi \
    --save-name forget \
    --resume-path ./exp_data/gmm/svi/full-train-ckpt-model.pkl \
    --cpu
