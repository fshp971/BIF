cd $3

for i in {1..5}
do
python main.py \
    --exp-name bnn \
    --exp-type burn-in-remain \
    --arch lenet \
    --dl-prior-sig 0.15 \
    --dataset fashion-mnist \
    --burn-in-steps 10000 \
    --optim sgld \
    --batch-size 128 \
    --lr 0.5 \
    --lr-decay-exp -0.5 \
    --ifs-scaling 0.005 \
    --ifs-iter-T 64 \
    --ifs-samp-T 5 \
    --ifs-iter-bs 128 \
    --ifs-rm-bs 64 \
    --ifs-kill-num $2 \
    --save-dir ./exp_data/fashion/sgld/$i/remain \
    --save-name remove-$1
done
