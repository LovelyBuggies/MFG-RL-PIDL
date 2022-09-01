import torch
import numpy as np
import os
import src.utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import time

from src.utils import save_dict_to_json, check_exist_and_create, check_and_make_dir
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from scipy.interpolate import griddata
import sys


def training_pre(model, optimizer, data_loader, device,
             physics=None,
             physics_optimizer=None,
             restore_from=None,
             epochs=1000,
             batch_size=None,
             experiment_dir=None,
             save_frequency=1,
             verbose_frequency=1,
             verbose_computation_time=0,
             save_each_epoch="False",
             ):
    begin_at_epoch = 0
    is_save = True
    writer = SummaryWriter(os.path.join(experiment_dir, "summary"))
    weights_path = os.path.join(experiment_dir, "weights")
    check_exist_and_create(weights_path)

    if restore_from is not None:
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch = utils.load_checkpoint(restore_from, model, optimizer, begin_at_epoch)
        if physics is not None:
            str_idx = restore_from.index('.tar')
            restore_from_physics=restore_from[:str_idx] + '.physics' + restore_from[str_idx:]
            assert os.path.isfile(restore_from_physics), "restore_from_physics is not a file"
            utils.load_checkpoint(restore_from_physics, physics, physics_optimizer, begin_at_epoch)
        logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")


    begin_at_epoch = 0

    best_loss = 10000
    best_last_train_loss = {"best":
                                {"loss": 100,
                                 "epoch": 0},
                            "last":
                                {"loss": 100,
                                 "epoch": 0},
                            }
    np.random.seed(1)
    for epoch in tqdm(range(begin_at_epoch, epochs)):


        num_steps = 10

        for step in range(num_steps):

            batch = data_loader.get_batch()
            x = batch["x_of_u"]
            t = batch["t_of_u"]
            u = batch["u"]
            X, T = np.meshgrid(x, t, indexing='ij')
            X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            X_star = torch.tensor(X_star, requires_grad=True).float().to(model.device)
            idx = np.random.choice(range(x.shape[0]*t.shape[0]), int(x.shape[0]*t.shape[0]*0.2), replace=False)
            u_star = u.flatten()[:, None]  # 960*240 by 1
            u_star = torch.tensor(u_star, requires_grad=True).float().to(model.device)
            phy_loss = torch.square(model.f(X_star[idx, :]) - u_star[idx, :])
            phy_loss = phy_loss.mean()*5000
            writer.add_scalar("loss/train", phy_loss, epoch + 1)
            optimizer.zero_grad()
            phy_loss.backward(retain_graph=True)
            optimizer.step()
            # loss_his.append(phy_loss.cpu().detach().numpy())

        # logging
        if verbose_frequency > 0:
            if epoch % verbose_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs}    loss={phy_loss:.3f}")

        # saving at every "save_frequency" or at the last epoch
        if (epoch % save_frequency == 0) | (epoch == begin_at_epoch + epochs - 1):
            is_save = True
            is_best = phy_loss < best_loss
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=weights_path,
                                  save_each_epoch=save_each_epoch)
            # if physics is not None:
            #     utils.save_checkpoint_physics({'epoch': epoch + 1,
            #                        'state_dict': physics.state_dict(),
            #                        'optim_dict': physics_optimizer.state_dict()},
            #                       is_best=is_best,
            #                       checkpoint=weights_path,
            #                       save_each_epoch=save_each_epoch)

            # if best loss, update the "best_last_train_loss"
            if is_best:
                best_loss = phy_loss
                best_last_train_loss["best"]["loss"] = best_loss
                best_last_train_loss["best"]["epoch"] = epoch+1

            # update and save the latest "best_last_train_loss"
            best_last_train_loss["last"]["loss"] = phy_loss
            best_last_train_loss["last"]["epoch"] = epoch+1

            save_path = os.path.join(experiment_dir, "best_last_train_loss.json")
            save_dict_to_json(best_last_train_loss, save_path)

            # save loss to tensorboard
            # writer.add_scalar("loss/train", loss_his[-1], epoch+1)

            # write the physics_params
            # for k, v in physics_params.items():
            #     writer.add_scalar(f"physics_params/{k:s}", v.mean(), epoch+1)


            # write the hist of the gradient w.r.t x and t
            # for k, v in grad_hist.items():
            #     writer.add_histogram(f"grad/{k:s}", v, epoch+1)

            # for k, v in physics.torch_meta_params.items():
            #     if physics.meta_params_trainable[k] == "True":
            #         writer.add_scalar(f"physics_grad/dLoss_d{k:s}", v.grad, epoch + 1)
        else:
            is_save=False

def test(model, data_loader, parameters,
         restore_from=None,
         physics=None,
         model_alias=None,
         save_dir=None
         ):
    if restore_from is not None:
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch = 0
        begin_at_epoch = utils.load_checkpoint(restore_from, model, optimizer=None, epoch=begin_at_epoch,
                                               device=model.device)
        logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")

    check_and_make_dir(os.path.join(save_dir, model_alias))
    save_path_metric = os.path.join(save_dir, model_alias,
                                    f"metrics_test.json")

    test_data = data_loader.get_test_data()

    # test for V, which is one-time-step wider than rho and u

    x = torch.tensor(test_data["x_of_u"], requires_grad=True).float().to(model.device)
    t = torch.tensor(test_data["t_of_u"], requires_grad=True).float().to(model.device)
    x = torch.unsqueeze(x, dim=-1)
    t = torch.unsqueeze(t, dim=-1)

    xdim = test_data["xdim_of_u"]
    tdim = test_data["tdim_of_u"]
    u_hat = model.f(torch.cat((x, t), 1))
    u_hat = u_hat.detach().cpu().numpy()
    # torch_params = self.sample_params(self.torch_meta_params, batch_size)
    u_hat = u_hat[:, 0].reshape(xdim, tdim)
    # u_hat = u_hat.detach().cpu().numpy().reshape(xdim, tdim)

    u = test_data["u"].reshape(xdim, tdim)
    # u = test_data["u"]

    metric_dict = dict()
    metric_dict["MSE_u"] = np.square(u - u_hat).mean()

    # save the metric
    save_dict_to_json(metric_dict, save_path_metric)

    # save the test result
    np.savetxt(os.path.join(save_dir, model_alias, f"u_hat.csv"), u_hat, delimiter=",")
    np.savetxt(os.path.join(save_dir, model_alias, f"u.csv"), u, delimiter=",")




