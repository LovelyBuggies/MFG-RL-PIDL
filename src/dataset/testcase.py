'''
Test case for the data_loader
Before the execution of this script, the working dir. should be set to the 'MFG-Traffic'
'''

### testcase for the sep

from lwr_sep import LwrSepLoader
from lwr_non_sep import LwrNonSepLoader

data_param = None

# train data
data_loader = LwrSepLoader(data_param)
batch = data_loader.get_batch()
print("========= sep =========")

print("train x_of_rho shape: ", batch["x_of_rho"].shape)
print("train t_of_rho shape: ", batch["t_of_rho"].shape)
print("train rho shape: ", batch["rho"].shape)
print("train x_of_V shape: ", batch["x_of_V"].shape)
print("train t_of_V shape: ", batch["t_of_V"].shape)
print("train V shape: ", batch["V"].shape)
print("train mean x:", batch["mean_x"])
print("train std x:", batch["std_x"])

# test data
test_data = data_loader.get_test_data()
print("test xdim_of_rho:", test_data["xdim_of_rho"])
print("test tdim_of_rho:", test_data["tdim_of_rho"])
print("test x_of_rho shape:", test_data["x_of_rho"].shape)
print("test t_of_rho shape:", test_data["t_of_rho"].shape)
print("test rho shape:", test_data["rho"].shape)

print("test xdim_of_V:", test_data["xdim_of_V"])
print("test tdim_of_V:", test_data["tdim_of_V"])
print("test x_of_V shape:", test_data["x_of_V"].shape)
print("test t_of_V shape:", test_data["t_of_V"].shape)
print("test V shape:", test_data["V"].shape)

print("test xdim_of_rho:", test_data["xdim_of_u"])
print("test tdim_of_rho:", test_data["tdim_of_u"])
print("test x_of_rho shape:", test_data["x_of_u"].shape)
print("test t_of_rho shape:", test_data["t_of_u"].shape)
print("test rho shape:", test_data["u"].shape)


### testcase for the non-sep
print("========= non-sep =========")

# train data
data_loader = LwrNonSepLoader(data_param)
batch = data_loader.get_batch()

print("train x_of_rho shape: ", batch["x_of_rho"].shape)
print("train t_of_rho shape: ", batch["t_of_rho"].shape)
print("train rho shape: ", batch["rho"].shape)
print("train x_of_V shape: ", batch["x_of_V"].shape)
print("train t_of_V shape: ", batch["t_of_V"].shape)
print("train V shape: ", batch["V"].shape)
print("train mean x:", batch["mean_x"])
print("train std x:", batch["std_x"])

# test data
test_data = data_loader.get_test_data()
print("test xdim_of_rho:", test_data["xdim_of_rho"])
print("test tdim_of_rho:", test_data["tdim_of_rho"])
print("test x_of_rho shape:", test_data["x_of_rho"].shape)
print("test t_of_rho shape:", test_data["t_of_rho"].shape)
print("test rho shape:", test_data["rho"].shape)

print("test xdim_of_V:", test_data["xdim_of_V"])
print("test tdim_of_V:", test_data["tdim_of_V"])
print("test x_of_V shape:", test_data["x_of_V"].shape)
print("test t_of_V shape:", test_data["t_of_V"].shape)
print("test V shape:", test_data["V"].shape)

print("test xdim_of_rho:", test_data["xdim_of_u"])
print("test tdim_of_rho:", test_data["tdim_of_u"])
print("test x_of_rho shape:", test_data["x_of_u"].shape)
print("test t_of_rho shape:", test_data["t_of_u"].shape)
print("test rho shape:", test_data["u"].shape)