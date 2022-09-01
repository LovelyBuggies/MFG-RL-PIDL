import torch
import math
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate


class MFG_sep(torch.nn.Module):

    def __init__(self, meta_params_value, lower_bounds, upper_bounds, params,
                 train=False,
                 device=None):
        super(MFG_sep, self).__init__()
        self.torch_meta_params = dict()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.train = train
        self.alpha = params.physics["alpha"]
        self.boundary = params.boundary_condition
        self.T = params.MFG["T"]
        self.umax = meta_params_value['umax']
        self.rhojam = meta_params_value['rhojam']
        self.L = params.MFG["L"]

    def caculate_residual(self, rho, V, x, t, Umax, RHOjam, model,
                                        epoch, writer, is_save):

        Umax = torch.tensor(Umax, requires_grad=False).float().to(model.device)
        RHOjam = torch.tensor(RHOjam, requires_grad=False).float().to(model.device)

        dV_dx = torch.autograd.grad(V, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]
        # tensor_min = torch.stack((Umax, Umax * (1 - Umax * torch.mean(dV_dx))))
        # min = torch.min(tensor_min)
        # u = torch.max(torch.stack((min, torch.tensor(0))))

        # u = torch.clamp(Umax * (1 - Umax * dV_dx), 0, Umax)
        u = torch.clamp(Umax * (1 - Umax * dV_dx), -5, 5)

        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0],1]).to(model.device), retain_graph=True)[0]
        drho_u_dx = torch.autograd.grad(rho*u, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]

        f1 = drho_dt + drho_u_dx
        f2 = dV_dt + u*dV_dx + 0.5*(u/Umax)**2 - u/Umax + rho/RHOjam

        if is_save:
            writer.add_histogram(f"grad/dV_dx", dV_dx, epoch + 1)
            writer.add_histogram(f"grad/u", u, epoch + 1)
            writer.add_histogram(f"grad/drho_dt", drho_dt, epoch + 1)
            writer.add_histogram(f"grad/drho_u_dx", drho_u_dx, epoch + 1)
            writer.add_histogram(f"grad/dV_dt", dV_dt, epoch + 1)

        return f1, f2

    def caculate_boundary_residual(self, model, batch):
        x_rho = torch.tensor(batch["x_of_rho"]).float().to(model.device)
        x_rho = torch.unsqueeze(x_rho, dim=-1)
        t_rho = torch.tensor(batch["t_of_rho"]).float().to(model.device)
        t_rho = torch.unsqueeze(t_rho, dim=-1)
        rho = torch.tensor(batch["rho"]).float().to(model.device)
        rho = torch.unsqueeze(rho, dim=-1)
        rho_hat = model.f(torch.cat((x_rho, t_rho), 1))[:,:1]
        fb1 = rho - rho_hat
        fb1 = torch.square(fb1)


        x_V =torch.tensor(batch["x_of_V"]).float().to(model.device)
        x_V = torch.unsqueeze(x_V, dim=-1)

        t_V = torch.tensor(batch["t_of_V"]).float().to(model.device)
        t_V = torch.unsqueeze(t_V, dim=-1)
        V = torch.tensor(batch["V"]).float().to(model.device)
        V = torch.unsqueeze(V, dim=-1)
        V_hat = model.f(torch.cat((x_V, t_V), 1))[:, 1:2]
        fb2 = V - V_hat
        fb2 = torch.square(fb2)

        t_boundary = torch.linspace(0, 1, 100)
        t_boundary = torch.unsqueeze(t_boundary, dim=-1)
        rho_boundary_start = model.f(torch.cat((torch.zeros_like(t_boundary), t_boundary), 1))[:,:1]
        rho_boundary_end = model.f(torch.cat((torch.full(t_boundary.shape, self.L), t_boundary), 1))[:,:1]

        fb3 = torch.square(rho_boundary_start - rho_boundary_end)

        return fb1, fb2, fb3

    def get_residuals(self, model, batch, epoch=0, writer=None, is_save=False):
        # get gradient
        x = np.linspace(0, 1, 20)
        t = np.linspace(0, 1, 20).flatten()[:,None]
        xx, tt = np.meshgrid(x, t, indexing='ij')
        xx = xx.flatten()[:,None]
        tt = tt.flatten()[:,None]
        idx = np.random.choice(xx.shape[0], 32, replace=False) ## hard code
        x_phy = xx[idx,:]
        t_phy = tt[idx,:]
        x = torch.tensor(x_phy, requires_grad=True).float().to(model.device)
        t = torch.tensor(t_phy, requires_grad=True).float().to(model.device)
        rho_V = model.f(torch.cat((x, t), 1))
        # torch_params = self.sample_params(self.torch_meta_params, batch_size)

        rho = torch.unsqueeze(rho_V[:, 0], dim=-1)
        V = torch.unsqueeze(rho_V[:, 1] , dim=-1)
        f1, f2 = self.caculate_residual(rho, V,  x, t, self.umax, self.rhojam, model,
                                        epoch, writer, is_save)

        f1 = torch.square(f1)
        f2 = torch.square(f2)

        fb_1, fb_2, fb_3 = self.caculate_boundary_residual(model, batch)

        gradient_hist = {"rho": rho.cpu().detach().numpy(),
                         "f1": f1.cpu().detach().numpy(),
                         "f2": f2.cpu().detach().numpy(),
                         "fb1": fb_1.cpu().detach().numpy(),
                         "fb2": fb_2.cpu().detach().numpy(),
                         "fb3": fb_3.cpu().detach().numpy(),
                         }

        # for k in torch_params.keys():
        #     torch_params[k] = torch_params[k].cpu().detach().numpy()

        r_final = self.alpha[0]*f1.mean() + self.alpha[1]*f2.mean() + \
                  self.alpha[2]*fb_1.mean() + self.alpha[3]*fb_2.mean() + self.alpha[3]*fb_3.mean()

        phy_loss = r_final

        if is_save:
            writer.add_scalar("loss/train", phy_loss, epoch + 1)
            writer.add_scalar("loss/f1", f1.mean(), epoch + 1)
            writer.add_scalar("loss/f2", f2.mean(), epoch + 1)
            writer.add_scalar("loss/fb_1", fb_1.mean(), epoch + 1)
            writer.add_scalar("loss/fb_2", fb_2.mean(), epoch + 1)
            writer.add_scalar("loss/fb_3", fb_3.mean(), epoch + 1)
            for k, v in gradient_hist.items():
                writer.add_histogram(f"grad/{k:s}", v, epoch+1)


        return phy_loss, gradient_hist


class MFG_nonsep(torch.nn.Module):

    def __init__(self, meta_params_value, lower_bounds, upper_bounds, params,
                 train=False,
                 device=None):
        super(MFG_nonsep, self).__init__()
        self.torch_meta_params = dict()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.train = train
        self.alpha = params.physics["alpha"]
        self.boundary = params.boundary_condition
        self.T = params.MFG["T"]
        self.umax = meta_params_value['umax']
        self.rhojam = meta_params_value['rhojam']
        self.L = params.MFG["L"]

    def caculate_residual(self, rho, V, x, t, Umax, RHOjam, model,
                          epoch, writer, is_save):

        Umax = torch.tensor(Umax, requires_grad=False).float().to(model.device)
        RHOjam = torch.tensor(RHOjam, requires_grad=False).float().to(model.device)

        dV_dx = torch.autograd.grad(V, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]
        # tensor_min = torch.stack((Umax, Umax * (1 - Umax * torch.mean(dV_dx))))
        # min = torch.min(tensor_min)
        # u = torch.max(torch.stack((min, torch.tensor(0))))

        u = torch.clamp(Umax * (1 - Umax * dV_dx - rho / RHOjam), 0, Umax)
        # u = torch.clamp(Umax * (1 - Umax * dV_dx - rho / RHOjam), -5, 5)

        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device), retain_graph=True)[0]
        drho_u_dx = \
        torch.autograd.grad(rho * u, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]

        Umax = torch.tensor(Umax, requires_grad=True).float().to(model.device)
        RHOjam = torch.tensor(RHOjam, requires_grad=True).float().to(model.device)

        f1 = drho_dt + drho_u_dx
        f2 = dV_dt + u * dV_dx + 0.5 * (u / Umax) ** 2 - u / Umax + u * rho / (Umax * RHOjam)

        if is_save:
            writer.add_histogram(f"grad/dV_dx", dV_dx, epoch + 1)
            writer.add_histogram(f"grad/u", u, epoch + 1)
            writer.add_histogram(f"grad/drho_dt", drho_dt, epoch + 1)
            writer.add_histogram(f"grad/drho_u_dx", drho_u_dx, epoch + 1)
            writer.add_histogram(f"grad/dV_dt", dV_dt, epoch + 1)

        return f1, f2

    def caculate_boundary_residual(self, model, batch):
        x_rho = torch.tensor(batch["x_of_rho"]).float().to(model.device)
        x_rho = torch.unsqueeze(x_rho, dim=-1)
        t_rho = torch.tensor(batch["t_of_rho"]).float().to(model.device)
        t_rho = torch.unsqueeze(t_rho, dim=-1)
        rho = torch.tensor(batch["rho"]).float().to(model.device)
        rho = torch.unsqueeze(rho, dim=-1)
        rho_hat = model.f(torch.cat((x_rho, t_rho), 1))[:, :1]
        fb1 = rho - rho_hat
        fb1 = torch.square(fb1)

        x_V = torch.tensor(batch["x_of_V"]).float().to(model.device)
        x_V = torch.unsqueeze(x_V, dim=-1)

        t_V = torch.tensor(batch["t_of_V"]).float().to(model.device)
        t_V = torch.unsqueeze(t_V, dim=-1)
        V = torch.tensor(batch["V"]).float().to(model.device)
        V = torch.unsqueeze(V, dim=-1)
        V_hat = model.f(torch.cat((x_V, t_V), 1))[:, 1:2]
        fb2 = V - V_hat
        fb2 = torch.square(fb2)

        t_boundary = torch.linspace(0, 1, 100)
        t_boundary = torch.unsqueeze(t_boundary, dim=-1)
        # for both rho and V??
        # rho_boundary_start = model.f(torch.cat((torch.zeros_like(t_boundary), t_boundary), 1))[:, :1]
        # rho_boundary_end = model.f(torch.cat((torch.full(t_boundary.shape, self.L), t_boundary), 1))[:, :1]
        rho_boundary_start = model.f(torch.cat((torch.zeros_like(t_boundary), t_boundary), 1))
        rho_boundary_end = model.f(torch.cat((torch.full(t_boundary.shape, self.L), t_boundary), 1))
        fb3 = torch.square(rho_boundary_start - rho_boundary_end)

        return fb1, fb2, fb3

    def get_residuals(self, model, batch, epoch=0, writer=None, is_save=False):
        # get gradient
        x = np.linspace(0, 1, 20)
        t = np.linspace(0, 1, 20).flatten()[:, None]
        xx, tt = np.meshgrid(x, t, indexing='ij')
        xx = xx.flatten()[:, None]
        tt = tt.flatten()[:, None]
        idx = np.random.choice(xx.shape[0], 256, replace=False)  ## hard code
        x_phy = xx[idx, :]
        t_phy = tt[idx, :]
        x = torch.tensor(x_phy, requires_grad=True).float().to(model.device)
        t = torch.tensor(t_phy, requires_grad=True).float().to(model.device)
        rho_V = model.f(torch.cat((x, t), 1))
        # torch_params = self.sample_params(self.torch_meta_params, batch_size)
        rho = torch.unsqueeze(rho_V[:, 0], dim=-1)
        V = torch.unsqueeze(rho_V[:, 1], dim=-1)
        f1, f2 = self.caculate_residual(rho, V, x, t, self.umax, self.rhojam, model,
                                        epoch, writer, is_save)

        f1 = torch.square(f1)
        f2 = torch.square(f2)

        fb_1, fb_2, fb_3 = self.caculate_boundary_residual(model, batch)

        gradient_hist = {"rho": rho.cpu().detach().numpy(),
                         "f1": f1.cpu().detach().numpy(),
                         "f2": f2.cpu().detach().numpy()*self.alpha[1],
                         "fb1": fb_1.cpu().detach().numpy()*self.alpha[2],
                         "fb2": fb_2.cpu().detach().numpy()*self.alpha[3],
                         "fb3": fb_3.cpu().detach().numpy()*self.alpha[4],
                         }

        # for k in torch_params.keys():
        #     torch_params[k] = torch_params[k].cpu().detach().numpy()

        r_final = self.alpha[0] * f1.mean() + self.alpha[1] * f2.mean() + \
                  self.alpha[2] * fb_1.mean() + self.alpha[3] * fb_2.mean() + self.alpha[4] * fb_3.mean()

        phy_loss = r_final

        if is_save:
            writer.add_scalar("loss/train", phy_loss, epoch + 1)
            writer.add_scalar("loss/f1", f1.mean()*self.alpha[0], epoch + 1)
            writer.add_scalar("loss/f2", f2.mean()*self.alpha[1], epoch + 1)
            writer.add_scalar("loss/fb_1", fb_1.mean()*self.alpha[2], epoch + 1)
            writer.add_scalar("loss/fb_2", fb_2.mean()*self.alpha[3], epoch + 1)
            writer.add_scalar("loss/fb_3", fb_3.mean()*self.alpha[4], epoch + 1)
            for k, v in gradient_hist.items():
                writer.add_histogram(f"grad/{k:s}", v, epoch + 1)

        return phy_loss, gradient_hist


class MFG_sep_fixu(torch.nn.Module):


    def __init__(self, meta_params_value, lower_bounds, upper_bounds, params,
                 train = False,
                 rho_model=None,
                 device=None):
        super(MFG_sep_fixu, self).__init__()
        self.torch_meta_params = dict()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.train = train
        self.alpha = params.physics["alpha"]
        self.boundary = params.boundary_condition
        self.T = params.MFG["T"]
        self.umax = meta_params_value['umax']
        self.rhojam = meta_params_value['rhojam']
        self.L = params.MFG["L"]
        self.model_rho = rho_model

    def caculate_residual(self, rho, V, u, du_dx, x, t, Umax, RHOjam, model,
                                        epoch, writer, is_save, u_net=None):

        Umax = torch.tensor(Umax, requires_grad=False).float().to(model.device)
        RHOjam = torch.tensor(RHOjam, requires_grad=False).float().to(model.device)
        # u = torch.tensor(u, requires_grad=False).float().to(model.device)

        u = u_net.f(torch.cat((x, t), 1))
        # u = 1-rho
        # du_dx = torch.tensor(du_dx, requires_grad=False).float().to(model.device)

        dV_dx = torch.autograd.grad(V, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]

        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device), retain_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, torch.ones([t.shape[0], 1]).to(model.device), retain_graph=True)[0]
        drho_u_dx = torch.autograd.grad(rho*u, x, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]
        drho_dx = torch.autograd.grad(rho, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        drho_dxx = torch.autograd.grad(drho_dx, x, torch.ones([x.shape[0], 1]).to(model.device),
                                       retain_graph=True)[0]

        dV_dt = torch.autograd.grad(V, t, torch.ones([x.shape[0], 1]).to(model.device), retain_graph=True)[0]

        # f1 = drho_dt + drho_u_dx
        # f1 = drho_dt + drho_u_dx - 0.005*drho_dxx
        f1 = drho_dt + drho_dx*u + rho*du_dx
        f2 = dV_dt + u*dV_dx + 0.5*(u/Umax)**2 - u/Umax + rho/RHOjam

        if is_save:
            writer.add_histogram(f"grad/dV_dx", dV_dx, epoch + 1)
            writer.add_histogram(f"grad/u", u, epoch + 1)
            writer.add_histogram(f"grad/drho_dt", drho_dt, epoch + 1)
            writer.add_histogram(f"grad/drho_u_dx", drho_u_dx, epoch + 1)
            writer.add_histogram(f"grad/dV_dt", dV_dt, epoch + 1)

        return f1, f2

    def caculate_boundary_residual(self, model, batch):

        x = batch['x_of_u']
        t = batch['t_of_u']
        xx, tt = np.meshgrid(x, t, indexing='ij')
        #[0, 50, 75, 100, 125,150, 175, 200, 220, 240]
        xx = torch.tensor(xx[[0, -1], :].flatten()[:, None]).float().to(model.device)
        tt = torch.tensor(tt[[0, -1], :].flatten()[:, None]).float().to(model.device)
        rho_label = torch.tensor(batch['rho'][[0, -1], :].flatten()[:, None]).float().to(model.device)
        rho_pre = model.f(torch.cat((xx, tt), 1))[:, :1]
        loss_loop = torch.square(rho_label-rho_pre)

        x_rho = torch.tensor(batch["x_of_rho"]).float().to(model.device)
        x_rho = torch.unsqueeze(x_rho, dim=-1)
        t_rho = torch.tensor(batch["t_of_rho"]).float().to(model.device)
        t_rho = torch.unsqueeze(t_rho, dim=-1)
        rho = torch.tensor(batch["rho_0"]).float().to(model.device)
        rho = torch.unsqueeze(rho, dim=-1)
        rho_hat = model.f(torch.cat((x_rho, torch.zeros_like(x_rho)), 1))[:, :1]
        fb1 = rho - rho_hat
        fb1 = torch.square(fb1)

        x_V = torch.tensor(batch["x_of_V"]).float().to(model.device)
        x_V = torch.unsqueeze(x_V, dim=-1)

        t_V = torch.tensor(batch["t_of_V"]).float().to(model.device)
        t_V = torch.unsqueeze(t_V, dim=-1)
        V = torch.tensor(batch["V"]).float().to(model.device)
        V = torch.unsqueeze(V, dim=-1)
        V_hat = model.f(torch.cat((x_V, torch.ones_like(x_V)), 1))[:, :2]
        fb2 = V - V_hat
        fb2 = torch.square(fb2)

        t_boundary = torch.linspace(0, 1, 100)
        t_boundary = torch.unsqueeze(t_boundary, dim=-1)
        rho_boundary_start = model.f(torch.cat((torch.zeros_like(t_boundary), t_boundary), 1))[:, :1]
        rho_boundary_end = model.f(torch.cat((torch.full(t_boundary.shape, self.L), t_boundary), 1))[:, :1]

        fb3 = torch.square(rho_boundary_start - rho_boundary_end)

        return fb1, fb2, fb3, loss_loop

    def get_residuals(self, model, batch, epoch=0, writer=None, is_save=False, u_net=None):
        # get gradient
        x = np.linspace(0, batch['x_of_rho'].max(), 50)
        t = np.linspace(0, batch['t_of_rho'].max(), 50)
        xx, tt = np.meshgrid(x, t, indexing='ij')
        xx = xx.flatten()[:, None]
        tt = tt.flatten()[:, None]
        number_of_p = 2500
        idx = np.random.choice(xx.shape[0], number_of_p, replace=False) ## hard code

        x_phy = xx[idx, :]
        t_phy = tt[idx, :]

        x_tensor = torch.tensor(x_phy, requires_grad=True).float().to(model.device)
        t_tensor = torch.tensor(t_phy, requires_grad=True).float().to(model.device)
        rho_V = model.f(torch.cat((x_tensor, t_tensor), 1))
        # torch_params = self.sample_params(self.torch_meta_params, batch_size)

        rho = torch.unsqueeze(rho_V[:, 0], dim=-1)
        V = torch.unsqueeze(rho_V[:, 1], dim=-1)

        # x_map, t_map = np.meshgrid(batch['x_of_u'], batch['t_of_u'])
        # f = interpolate.interp2d(batch['x_of_u'], batch['t_of_u'], batch['u'], kind='cubic')
        # f_du_dx = interpolate.interp2d(batch['x_of_u'], batch['t_of_u'], batch['du_dx'], kind='cubic')

        list_u = []
        # for i in range(number_of_p):
        #     list_u.append(f(x_phy.squeeze()[i], t_phy.squeeze()[i]))
        u = np.array(list_u)

        list_du_dx = []
        # for i in range(number_of_p):
        #     list_du_dx.append(f_du_dx(x_phy.squeeze()[i], t_phy.squeeze()[i]))
        du_dx = np.array(list_du_dx)

        f1, f2 = self.caculate_residual(rho, V, u, du_dx, x_tensor, t_tensor, self.umax, self.rhojam, model,
                                        epoch, writer, is_save, u_net=u_net)

        f1 = torch.square(f1)
        f2 = torch.square(f2)

        fb_1, fb_2, fb_3, loss_loop = self.caculate_boundary_residual(model, batch)

        gradient_hist = {"rho": rho.cpu().detach().numpy(),
                         "f1": f1.cpu().detach().numpy(),
                         "f2": f2.cpu().detach().numpy(),
                         "fb1": fb_1.cpu().detach().numpy(),
                         "fb2": fb_2.cpu().detach().numpy(),
                         "fb3": fb_3.cpu().detach().numpy(),
                         }

        # for k in torch_params.keys():
        #     torch_params[k] = torch_params[k].cpu().detach().numpy()

        r_final = self.alpha[0]*f1.mean() + self.alpha[1]*f2.mean() + \
                  self.alpha[2]*fb_1.mean() + self.alpha[3]*fb_2.mean() + self.alpha[4]*fb_3.mean() + self.alpha[5]*loss_loop.mean()

        phy_loss = r_final

        if is_save:
            writer.add_scalar("loss/train", phy_loss, epoch + 1)
            writer.add_scalar("loss/f1", f1.mean()*self.alpha[0], epoch + 1)
            writer.add_scalar("loss/f2", f2.mean()*self.alpha[1], epoch + 1)
            writer.add_scalar("loss/fb_1", fb_1.mean()*self.alpha[2], epoch + 1)
            writer.add_scalar("loss/fb_2", fb_2.mean()*self.alpha[3], epoch + 1)
            writer.add_scalar("loss/fb_3", fb_3.mean()*self.alpha[4], epoch + 1)
            writer.add_scalar("loss/loss_loop", loss_loop.mean()*self.alpha[5], epoch + 1)
            for k, v in gradient_hist.items():
                writer.add_histogram(f"grad/{k:s}", v, epoch+1)


        return phy_loss, gradient_hist




