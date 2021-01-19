import argparse
import pickle
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None,
        choices=['svi', 'sgld', 'sghmc'])

    return parser.parse_args()


def main():
    args = get_args()

    exp_num = 5

    save_db = dict()

    ''' pd: processed; npd: non-processed; sc: scratch '''
    for name in ['pd', 'npd', 'sc']:

        res_err, del_err, test_err = [], [], []

        for kk in ['1k', '2k', '3k', '4k', '5k', '6k']:
            res_e, del_e, test_e = [], [], []
            for ii in range(1, exp_num+1):
                if name == 'sc':
                    path = './{}/{}/remain/remove-{}-ckpt-log.pkl'.format(args.name, ii, kk)
                    with open(path, 'rb') as f:
                        dat = pickle.load(f)
                    res_e.append(1. - dat['(remain) train_acc'][-1])
                    del_e.append(1. - dat['forgetted_train_acc'][-1])
                    test_e.append(1. - dat['test_acc'][-1])

                else:
                    path = './{}/{}/forget/forget-{}-ckpt-log.pkl'.format(args.name, ii, kk)
                    with open(path, 'rb') as f:
                        dat = pickle.load(f)
                    idx = -1
                    if name == 'npd': idx = 0
                    res_e.append(1. - dat['remain_train_acc'][idx])
                    del_e.append(1. - dat['forgetted_train_acc'][idx])
                    test_e.append(1. - dat['test_acc'][idx])

            res_err.append(res_e)
            del_err.append(del_e)
            test_err.append(test_e)

        res_err = np.array(res_err)
        del_err = np.array(del_err)
        test_err = np.array(test_err)

        save_db[name] = dict()

        save_db[name]['res_err'] = res_err.mean(axis=1)
        save_db[name]['del_err'] = del_err.mean(axis=1)
        save_db[name]['test_err'] = test_err.mean(axis=1)

        cof = exp_num / (exp_num-1)
        save_db[name]['res_err_std'] = res_err.std(axis=1) * cof
        save_db[name]['del_err_std'] = del_err.std(axis=1) * cof
        save_db[name]['test_err_std'] = test_err.std(axis=1) * cof

    with open('save-db-{}.pkl'.format(args.name), 'wb') as f:
        pickle.dump(save_db, f)

    return


if __name__ == '__main__':
    main()

