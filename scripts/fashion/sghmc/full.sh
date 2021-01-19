cd $1

for i in {1..5}
do
python main.py \
    --exp-name bnn \
    --exp-type burn-in-remain \
    --arch lenet \
    --dl-prior-sig 0.15 \
    --dataset fashion-mnist \
    --burn-in-steps 10000 \
    --optim sghmc \
    --batch-size 128 \
    --lr 0.5 \
    --lr-decay-exp -0.5 \
    --ifs-scaling 0.05 \
    --ifs-iter-T 64 \
    --ifs-samp-T 5 \
    --ifs-iter-bs 128 \
    --ifs-rm-bs 64 \
    --ifs-kill-num 0 \
    --sghmc-alpha 0.4 \
    --save-dir ./exp_data/fashion/sghmc/$i \
    --save-name full-train
done
