import torch
import numpy as np
from src.dataset.lwr_origin import LwrLoader
from src.dataset.lwr_sep import LwrSepLoader, LwrSepLoaderFullObs
from src.dataset.lwr_flatten_initial import Lwr_f_Loader
from src.utils import Params, save_checkpoint, load_checkpoint
from src.utils import set_logger, delete_file_or_folder
from src.model.u_net import u_net
from src.training_pre import training_pre, test
import argparse
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments_pre/test_rho_net', # lwr_with_u_learning_z
                    help="Directory containing experiment_setting.json")
parser.add_argument('--mode', default='test',
                    help="train, test, or train_and_test")
parser.add_argument('--force_overwrite', default=True, action='store_true',
                    help="For debug. Force to overwrite")
parser.add_argument('--restore_from', default= None, #"experiments/lwr_learning_z/weights/last.path.tar",
                    help="Optional, file location containing weights to reload")



if torch.cuda.is_available():

    device = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info("Enable cuda")
else:
    device = torch.device('cpu')
    logging.info("cuda is not available")

if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'experiment_setting.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    net_args = (params.affine_coupling_layers_u["net"]["in_dim"],
                params.affine_coupling_layers_u["net"]["out_dim"],
                params.affine_coupling_layers_u["net"]["n_hidden"],
                params.affine_coupling_layers_u["net"]["hidden_dim"])

    net_kwargs = {"activation_type": params.affine_coupling_layers_u["net"]["activation_type"],
                  "last_activation_type": params.affine_coupling_layers_u["net"]["last_activation_type"],
                  "device": device}

    data_loader = Lwr_f_Loader()
    # data_loader = LwrLoader()
    model = u_net(data_loader,
                    device,
                    net_args,
                    net_kwargs)

    model.to(device)

    if params.affine_coupling_layers_u["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True]
                                     , **params.affine_coupling_layers_u["optimizer"]["kwargs"])
        # optimizer = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad == True],
        #                               history_size = 10000,
        #                               max_iter=1000)
        optimizer.param_groups[0]['capturable'] = True

    if args.mode == "train":
        training_pre(model, optimizer, data_loader, device,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epochs,
                 experiment_dir=args.experiment_dir,
                 save_frequency=params.save_frequency,
                 verbose_frequency=params.verbose_frequency,
                 save_each_epoch=params.save_each_epoch,
                 verbose_computation_time=params.verbose_computation_time
                 )

    elif args.mode == "test":
        save_dir = os.path.join(args.experiment_dir, "test_result/")
        # using last weights
        restore_from = os.path.join(args.experiment_dir, "weights/last.path.tar")
        model_alias = "last"
        test(model, data_loader, params,
             restore_from = restore_from,
             save_dir = save_dir,
             model_alias = model_alias)

        # using best weights
        restore_from = os.path.join(args.experiment_dir, "weights/best.pth.tar")
        model_alias = "best"
        test(model, data_loader, params,
             restore_from = restore_from,
             save_dir = save_dir,
             model_alias = model_alias,
             )

