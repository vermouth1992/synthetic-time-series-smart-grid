"""
Train Recurrent GAN on Pecan dataset user 93
"""
from __future__ import print_function, division


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for training ACGAN on pecan street dataset')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--id', type=int, default=93)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())

    import numpy as np
    from data import preprocess_pecan_dataset, load_pecan_dataset
    from acgan import ACGAN as Model
    import pprint

    pprint.pprint(args)

    train = args['train']
    user_id_pv = args['id']

    all_user_id_pv = load_pecan_dataset()[0].tolist()
    if user_id_pv:
        assert user_id_pv in all_user_id_pv, 'Invalid id {}. Must be within {}'.format(user_id_pv, all_user_id_pv)
        user_id_pv = [user_id_pv]
    else:
        user_id_pv = all_user_id_pv

    for user_id in user_id_pv:
        print('Starting training model for user {}'.format(user_id))
        (usage, gen), (usage_recover, gen_recover), (month_label, day_label) = preprocess_pecan_dataset(user_id=user_id,
                                                                                                        threshold=(
                                                                                                            -2, 2))
        usage = np.expand_dims(usage, axis=-1)
        gen = np.expand_dims(gen, axis=-1)
        x = np.concatenate((usage, gen), axis=-1)
        num_train = 365 * 3
        x_train = x[:num_train]
        x_val = x[num_train:]
        month_label_train = month_label[:num_train]
        month_label_val = month_label[num_train:]
        day_label_train = day_label[:num_train]
        day_label_val = day_label[num_train:]
        print(x_train.shape, x_val.shape, month_label_train.shape, month_label_val.shape, day_label_train.shape,
              day_label_val.shape)

        weight_path = 'weights/pecan' + '_user_' + str(user_id) + '_'
        model = Model(input_dim=2, window_length=96, weight_path=weight_path)

        if train:
            num_epoch = args['num_epoch']
            print(model.discriminator.summary())
            print(model.generator.summary())
            model.train([x_train, month_label_train, day_label_train], [x_val, month_label_val, day_label_val],
                        num_epoch=num_epoch)
        else:
            x_generated = model.generate_by_date(1461)
            usage_generated = x_generated[:, :, 0]
            gen_generated = x_generated[:, :, 1]
            usage_generated_recover = usage_recover(usage_generated)
            gen_generated_recover = gen_recover(gen_generated)
            data = np.stack((usage_generated_recover, gen_generated_recover), axis=-1)
            print('The shape of the synthetic data is {}'.format(data.shape))
            np.savez_compressed('synthetic/user_' + str(user_id) + '_acgan.npz', data=data)
