# Tutorial of Reproducing the Experiments

This folder contains all the scripts used to conduct the experiments in our paper. Here, we give a tutorial on how to reproduce these experiments.

## Code Structure

```
|----./
    |---- experiments/
        |---- __init__.py
        |---- based_experiment.py
        |---- gmm_simulation.py
        |---- deep_learning.py
    |---- bayes_forgetters/
    	|---- __init__.py
        |---- bif_forgetter.py
        |---- sgmcmc_forgetter.py
        |---- vi_forgetter.py
    |---- models/
        |---- bayes_nn/
            |---- __init__.py
            |---- mcmc_modules.py
            |---- normal_modules.py
        |---- __init__.py
        |---- gmm.py
        |---- mcmc_models.py
        |---- normal_models.py
    |---- utils/
        |---- datasets/
            |---- __init__.py
            |---- gmm_datasets.py
            |---- torchvision_datasets.py
        |---- sgmcmc_optim/
            |---- __init__.py
            |---- sghmc.py
            |---- sgld.py
        |---- __init__.py
        |---- argument.py
        |---- data.py
        |---- generic.py
    |---- extract_bnn_fashion_data.py
    |---- gen_simulation_data.py
    |---- main.py
```

## Gaussian Mixture Model on Synthetic Dataset

The scripts for the Gaussian mixture model (GMM) experiments is as follows:

```
|---- ./
    |---- scripts/
        |---- gmm/
            |---- gmm-svi-full.sh
            |---- gmm-svi-remain.sh
            |---- gmm-svi-forget.sh
            |---- gmm-sgld-full.sh
            |---- gmm-sgld-remain.sh
            |---- gmm-sgld-forget.sh
            |---- gmm-sghmc-full.sh
            |---- gmm-sghmc-remain.sh
            |---- gmm-sghmc-forget.sh
```

### Datasets

The synthetic dataset is a two-dimensional real-valued dataset. It is at `./data/GMMs/gmm-2d-syn-set.pkl`. You can also generate it with the following command:

```bash
python gen_simulation_data.py --type=special
```

(Note that you may not be able to generate the same dataset used in our paper. This is because the random algorithm in `NumPy` may vary from different experiment environments.)

### Run the experiments with scripts

To conduct the experiment of variational inference, run the following commands:

```bash
# train on the full set
bash ./scripts/gmm/gmm-svi-full.sh ./

# train on the remaining set
bash ./scripts/gmm/gmm-svi-remain.sh ./

# perform forgetting for the model that trained on the full set
bash ./scripts/gmm/gmm-svi-forget.sh ./
```

Note that you need to run the script `gmm-svi-full.sh` first before running the script `gmm-svi-forget.sh`. Once finished, the experiment result of variational inference will be saved in `./exp_data/gmm/svi/`.

You can run the experiments of SGLD and SGHMC follow similar steps.

### Visualize the experiments results

See `./notebook/gmm-svi.ipynb` and `./notebook/gmm-mcmc.ipynb` for details.

## Bayesian Neural Network on Fashion-MNIST

The scripts for the Bayesian neural network (BNN) experiments is as follows:

```
|---- ./
    |---- scripts/
        |---- fashion/
            |---- svi/
                |---- full.sh
                |---- rm-xk.sh
                |---- forget.sh
            |---- sgld/
                ....
            |---- sghmc/
                ....
```

### Run the experiments with scripts

We take the experiment for variational BNN as an example.

To train the variational BNN on the full training set, run the following command:

```bash
bash ./scripts/fashion/svi/full.sh ./
```

To train the variational BNN on different remaining sets, run the following commands:

```bash
bash ./scripts/fashion/svi/rm-xk.sh 1k 1000 ./
bash ./scripts/fashion/svi/rm-xk.sh 2k 2000 ./
bash ./scripts/fashion/svi/rm-xk.sh 3k 3000 ./
bash ./scripts/fashion/svi/rm-xk.sh 4k 4000 ./
bash ./scripts/fashion/svi/rm-xk.sh 5k 5000 ./
bash ./scripts/fashion/svi/rm-xk.sh 6k 6000 ./
```

where the first parameter of the script `rm-xk.sh` specifies **the saving name**, and the second parameter specifies **the number of the removed datums**.

To conduct forgetting for the variational BNN, run the following commands:

```bash
bash ./scripts/fashion/svi/forget.sh 1k 1000 ./
bash ./scripts/fashion/svi/forget.sh 2k 2000 ./
bash ./scripts/fashion/svi/forget.sh 3k 3000 ./
bash ./scripts/fashion/svi/forget.sh 4k 4000 ./
bash ./scripts/fashion/svi/forget.sh 5k 5000 ./
bash ./scripts/fashion/svi/forget.sh 6k 6000 ./
```

where the first parameter of the script `forget.sh` specifies **the saving name**, and the second parameter specifies **the number of the datums that going to be removed**. Note that you need to run the script `full.sh` first before running the script `forget.sh`.

Once finished, the experiment result of variational BNN will be saved in `./exp_data/fashion/svi/`.

You can run the experiments of MCMC BNN with SGLD and SGHMC follow similar steps.

### Visualize the experiments results

First, copy the code for extracting data to the target location and change the working directory:  

```bash
cp ./extract_bnn_fashion_data.py ./exp_data/fashion/
cd ./exp_data/fashion/
```

Then, extract the experiments data with the following commands:

```bash
python extract_bnn_fashion_data.py --name svi
python extract_bnn_fashion_data.py --name sgld
python extract_bnn_fashion_data.py --name sghmc
```

The extracted experiments data will be saved in `./exp_data/fashion/save-db-svi.pkl`, `./exp_data/fashion/save-db-sgld.pkl` and `./exp_data/fashion/save-db-sghmc.pkl`.

Finally, see `./notebook/bnn-fashion-all.ipynb` for details.