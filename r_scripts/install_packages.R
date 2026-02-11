# Install packages

# Torch
# 1. Force R to use the Posit Package Manager binary for specific Linux version
options(repos = c(RSPM = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))
# 2. install torch
install.packages("torch")
# 3. Load and reinstall torch
torch::install_torch()


# Other packages
install.packages("luz")
install.packages("tidyverse")
install.packages("tidymodels")
install.packages("MLDataR")
install.packages("tabnet")
install.packages("vip")
install.packages("innsight")
install.packages("skimr")
