cd $1

python main.py \
    --exp-name gmm \
    --exp-type burn-in-remain \
    --gmm-kk 4 \
    --gmm-std 1 \
    --dataset gmm2d \
    --optim sghmc \
    --batch-size 64 \
    --burn-in-steps 2000 \
    --eval-freq 100 \
    --lr 2 \
    --lr-decay-exp -0.15 \
    --ifs-scaling 1 \
    --ifs-iter-T 32 \
    --ifs-samp-T 5 \
    --ifs-iter-bs 64 \
    --ifs-rm-bs 4 \
    --ifs-kill-num 0 \
    --sghmc-alpha 0.4 \
    --mcmc-samp-num 500 \
    --save-dir ./exp_data/gmm/sghmc \
    --save-name full-train \
    --cpu
