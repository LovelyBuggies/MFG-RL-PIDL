import argparse
import logging
import os
import sys
import src.utils as utils
import numpy as np
import torch
from src.utils import Params, save_checkpoint, load_checkpoint
from src.utils import set_logger, delete_file_or_folder
from src.layers.physics import MFG_sep, MFG_nonsep, MFG_sep_fixu
from src.model.MFG_net import MFG_net, MFG_net_2output
from value_iteration_ddpg import Actor
from src.training import training, test

from src.dataset.lwr_non_sep import LwrNonSepLoader, LwrNonSepLoaderFullObs
from src.dataset.lwr_sep import LwrSepLoader, LwrSepLoaderFullObs
from src.dataset.lwr_flatten_initial import Lwr_f_Loader


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/test_case_lwr_0.5_0.6', #lwr_with_u_learning_z
                    help="Directory containing experiment_setting.json")
parser.add_argument('--mode', default='train',
                    help="train, test, or train_and_test")
parser.add_argument('--force_overwrite', default=True, action='store_true',
                    help="For debug. Force to overwrite")
parser.add_argument('--restore_from', default= None, #"experiments/lwr_learning_z/weights/last.path.tar",
                    help="Optional, file location containing weights to reload")

# Set the random seed for the whole graph for reproductible experiments
def get_rho_network(u_network_path):
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'experiment_setting.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    torch.autograd.set_detect_anomaly(True)
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logging.info("Enable cuda")
    else:
        device = torch.device('cpu')
        logging.info("cuda is not available")

    if args.mode == "test":
        device = torch.device('cpu')
        logging.info("In the test mode, use cpu")

    # Safe Overwrite. Avoid to overwrite the previous experiment by mistake.
    force_overwrite = args.force_overwrite
    if force_overwrite is True:
        safe_files = ["experiment_setting.json", "safe_folder"]
        if args.restore_from is not None:
            safe_files.append(os.path.split(args.restore_from)[-2])

        if args.mode == "test":
            # every file under the root of the "experiment_dir"
            for file_folder in os.listdir(args.experiment_dir):
                if file_folder != "test_result":
                    safe_files.append(file_folder)

        # delete everything that is not in "safe_files"
        for file_folder in os.listdir(args.experiment_dir):
            if file_folder not in safe_files:
                delete_file_or_folder(os.path.join(args.experiment_dir, file_folder))
    # Set the logger
    set_logger(os.path.join(args.experiment_dir, 'train.log'))
    logging.info(" ".join(sys.argv))

    # Create the input data pipeline
    logging.info("Loading the datasets...")



    ## load data TODO
    if params.data["type"] == "lwr_non_sep":
        data_loader = LwrNonSepLoader(params.data["param"])
    elif params.data["type"] == "lwr_sep":
        data_loader = LwrSepLoader(params.data["param"])
    elif params.data["type"] == "lwr_non_sep_full":
        data_loader = LwrNonSepLoaderFullObs(params.data["param"])
    elif params.data["type"] == "lwr_sep_full":
        data_loader = LwrSepLoaderFullObs(params.data["param"])
    elif params.data["type"] == "lwr_flatten":
        data_loader = Lwr_f_Loader()
    else:
        raise ValueError("non-valid data type")
    ## load parameters TODO


    net_args = (params.affine_coupling_layers["net"]["in_dim"],
                params.affine_coupling_layers["net"]["out_dim"],
                params.affine_coupling_layers["net"]["n_hidden"],
                params.affine_coupling_layers["net"]["hidden_dim"])

    net_kwargs = {"activation_type": params.affine_coupling_layers["net"]["activation_type"],
                      "last_activation_type": params.affine_coupling_layers["net"]["last_activation_type"],
                      "device": device}


    ## load Physics
    if params.physics["type"] == "MFG_sep":
        physics = MFG_sep(params.physics["meta_params_value"],
                              params.physics["lower_bounds"],
                              params.physics["upper_bounds"],
                              params,
                              train=(params.physics["train"] == "True"))
        physics.to(device)
    if params.physics["type"] == "MFG_nonsep":
        physics = MFG_nonsep(params.physics["meta_params_value"],
                              params.physics["lower_bounds"],
                              params.physics["upper_bounds"],
                              params,
                              train=(params.physics["train"] == "True"))
        physics.to(device)

    if params.physics["type"] == "MFG_sep_fixu":
        physics = MFG_sep_fixu(params.physics["meta_params_value"],
                             params.physics["lower_bounds"],
                             params.physics["upper_bounds"],
                             params,
                             train=(params.physics["train"] == "True"))
        physics.to(device)

    # if params.physics["type"] == "MFG_nonsep_two_boundary":
    #     physics = MFG_nonsep_two_boundary(params.physics["meta_params_value"],
    #                           params.physics["lower_bounds"],
    #                           params.physics["upper_bounds"],
    #                           params,
    #                           train=(params.physics["train"] == "True"))
    #     physics.to(device)
    # elif params.physics["type"] == "non_sep":
    #     physics = MFG_nonsep(params.physics["meta_params_value"],
    #                           params.physics["meta_params_trainable"],
    #                           params.physics["lower_bounds"],
    #                           params.physics["upper_bounds"],
    #                           params.physics["hypers"],
    #                           train=(params.physics["train"] == "True"))
    #     physics.to(device)
    # if physics is not None:
    #     if params.physics["optimizer"]["type"] == "Adam":
    #         optimizer_physics = torch.optim.Adam(
    #             [p for p in physics.torch_meta_params.values() if p.requires_grad == True]
    #             , **params.physics["optimizer"]["kwargs"])
    #     elif params.physics["optimizer"]["type"] == "SGD":
    #         optimizer_physics = torch.optim.SGD(
    #             [p for p in physics.torch_meta_params.values() if p.requires_grad == True]
    #             , **params.physics["optimizer"]["kwargs"])
    #     elif params.physics["optimizer"]["type"] == "none":
    #         optimizer_physics = None
    # else:
    #     optimizer_physics = None


    ## load model

    if params.affine_coupling_layers["net"]["type"] == "double":
        model = MFG_net_2output(params.affine_coupling_layers["train"], data_loader,
                           device,
                           net_args,
                           net_kwargs)
    elif params.affine_coupling_layers["net"]["type"] == "single":
        model = MFG_net(params.affine_coupling_layers["train"], data_loader,
                           device,
                           net_args,
                           net_kwargs)
    else:
        raise ValueError("non-valid model type")
    model.to(device)



    net_args_u = (params.affine_coupling_layers_u["net"]["in_dim"],
                params.affine_coupling_layers_u["net"]["out_dim"],
                params.affine_coupling_layers_u["net"]["n_hidden"],
                params.affine_coupling_layers_u["net"]["hidden_dim"])

    net_kwargs_u = {"activation_type": params.affine_coupling_layers_u["net"]["activation_type"],
                  "last_activation_type": params.affine_coupling_layers_u["net"]["last_activation_type"],
                  "device": device}
    model_u = Actor(2)
    model_u.to(device)
    begin_at_epoch = 0
    restore_from = u_network_path
    tmp = torch.load(restore_from)
    begin_at_epoch = model_u.load_state_dict(tmp)
    # logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")


    if params.affine_coupling_layers["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True]
                                     , **params.affine_coupling_layers["optimizer"]["kwargs"])
        # optimizer = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad == True],
        #                               history_size = 10000,
        #                               max_iter=1000)
        optimizer.param_groups[0]['capturable'] = True

    if args.mode == "train":
        training(model, optimizer, data_loader, device,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epochs,
                 physics=physics,
                 experiment_dir=args.experiment_dir,
                 save_frequency=params.save_frequency,
                 verbose_frequency=params.verbose_frequency,
                 save_each_epoch=params.save_each_epoch,
                 verbose_computation_time=params.verbose_computation_time,
                 u_net=model_u
                 )
    elif args.mode == "test":
        save_dir = os.path.join(args.experiment_dir, "test_result/")
        # using last weights
        restore_from = os.path.join(args.experiment_dir, "weights/last.path.tar")
        model_alias = "last"
        test(model, data_loader, params,
             restore_from = restore_from,
             physics = physics,
             save_dir = save_dir,
             model_alias = model_alias)

        # using best weights
        restore_from = os.path.join(args.experiment_dir, "weights/best.pth.tar")
        model_alias = "best"
        test(model, data_loader, params,
             restore_from = restore_from,
             physics = physics,
             save_dir = save_dir,
             model_alias = model_alias,
             )

    else:
        raise ValueError("args.mode invalid")

    return model
