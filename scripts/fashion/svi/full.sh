cd $1

for i in {1..5}
do
python main.py \
    --exp-name bnn \
    --exp-type burn-in-remain \
    --is-vi \
    --dl-bbp-T 5 \
    --arch lenet \
    --dl-prior-sig 0.15 \
    --dataset fashion-mnist \
    --burn-in-steps 10000 \
    --optim sgd \
    --batch-size 128 \
    --lr 0.5 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 4000 \
    --ifs-scaling 0.1 \
    --ifs-iter-T 64 \
    --ifs-iter-bs 128 \
    --ifs-rm-bs 64 \
    --ifs-kill-num 0 \
    --save-dir ./exp_data/fashion/svi/$i \
    --save-name full-train
done
