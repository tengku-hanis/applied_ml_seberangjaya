##=============================================================================##
## Title: Deep neural network for classification - intermediate using torch/luz
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: February 12, 2026
##=============================================================================##

# DNN using torch/luz
# Using dataset()
# Using callbacks

# Apply XAI

# Packages ----------------------------------------------------------------

library(torch)
library(luz)
library(tidyverse)
library(tidymodels)
library(MLDataR)
library(innsight)


# Data --------------------------------------------------------------------

heart_df <- 
  heartdisease  %>% 
  mutate(across(c(Sex, RestingECG, Angina), as.factor))


# Dataset function -------------------------------------------------------

heart_dataset <- dataset(
  initialize = function(df) {
    # Pre-process and store as tensors
    self$x_num <- df %>% 
      select(Age, RestingBP, Cholesterol, FastingBS, MaxHR, HeartPeakReading) %>% 
      mutate(across(everything(), scale)) %>% 
      as.matrix() %>% torch_tensor(dtype = torch_float())
    
    self$x_cat <- model.matrix(~ Sex + RestingECG + Angina, data = df)[, -1] %>% 
      as.matrix() %>% torch_tensor(dtype = torch_float())
    
    self$y <- torch_tensor(as.matrix(df$HeartDisease), dtype = torch_float())
  },
  .getitem = function(i) {
    # 2. Simply return the slice
    list(x = list(self$x_num[i, ], self$x_cat[i, ]), y = self$y[i])      
  },
  .length = function() {
    self$y$size(1)
  }
)

# Convert to torch dataset
ds_tensor <- heart_dataset(heart_df)
ds_tensor[1]


# Dataloader -------------------------------------------------------------

set.seed(123) 
n <- nrow(heart_df)
train_size <- floor(0.6 * n)
valid_size <- floor(0.2 * n)

# Create indices
all_indices <- 1:n
train_indices <- sample(all_indices, size = train_size)

remaining_indices <- setdiff(all_indices, train_indices)
valid_indices <- sample(remaining_indices, size = valid_size)

test_indices <- setdiff(remaining_indices, valid_indices)

# Create Subsets
train_ds <- dataset_subset(ds_tensor, train_indices)
valid_ds <- dataset_subset(ds_tensor, valid_indices)
test_ds  <- dataset_subset(ds_tensor, test_indices)

# Create Dataloaders
train_dl <- train_ds %>% 
  dataloader(batch_size = 10, shuffle = TRUE)

valid_dl <- valid_ds %>% 
  dataloader(batch_size = 10, shuffle = FALSE)

test_dl <- test_ds %>% 
  dataloader(batch_size = 10, shuffle = FALSE)


# Specify the model ------------------------------------------------------

net <- 
  nn_module(
    initialize = function(d_in){
      self$net <- nn_sequential(
        nn_linear(d_in, 32),
        nn_relu(),
        nn_dropout(0.5),
        nn_linear(32, 64),
        nn_relu(),
        nn_dropout(0.5),
        nn_linear(64, 1),
        nn_sigmoid()
      )
    },
    forward = function(x){
      # x is currently a list
      # We must concatenate them along the feature dimension (dim=2)
      input <- torch_cat(x, dim = 2)
      self$net(input)
    }
  )


# Fit the model ----------------------------------------------------------

# Set parameters
d_in <- length(ds_tensor[1]$x[[1]]) + length(ds_tensor[1]$x[[2]]) # no of features minus the outcome

# Fit
fitted <- 
  net %>% 
  setup(
    loss = nn_bce_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_binary_accuracy(), luz_metric_binary_auroc())
  ) %>% 
  set_hparams(d_in = d_in) %>% 
  fit(
    train_dl, 
    epoch = 50, 
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_early_stopping(patience = 10),
      luz_callback_keep_best_model()
    )
  )

# XAI --------------------------------------------------------------------

# Extract the sequential model
model <- fitted$model$net$cpu()

# Define input and output names
input_names <- c(
  # Numeric variables
  "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "HeartPeakReading",
  # Categorical dummies (from model.matrix ~ .-1)
  "Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"
)

output_names <- c("Probability of heart disease")

# Create the Converter
# input_dim is 10 (6 numeric + 4 categorical)
converter <- convert(
  model,
  input_dim = 10,
  input_names = input_names,
  output_names = output_names
)

# Manually extract and concatenate the test data to match 
# what the 'model_core' expects (a single matrix of shape [N, 10])
idxs <- test_ds$indices
x_num <- ds_tensor$x_num[idxs, ]
x_cat <- ds_tensor$x_cat[idxs, ]

# Combine into one tensor and convert to R array
input_tensor <- torch_cat(list(x_num, x_cat), dim = 2)
input_data <- as_array(input_tensor)

# Run LRP (Layer-wise Relevance Propagation)
# Using 'alpha_beta' rule which is common for neural networks
lrp_result <- run_lrp(converter, input_data, rule_name = "alpha_beta", rule_param = 1)

# Check dimensions of result (Instances x Features x Outputs)
dim(get_result(lrp_result))

# Individual plots for the first two test instances
plot(lrp_result, data_idx = c(1, 2)) +
  theme_bw() +
  coord_flip() +
  labs(title = "Individual Feature Relevance (Patient 1 & 2)")

# Global boxplot - the overall importance of features across the entire test set
boxplot(lrp_result) +
  theme_bw() +
  coord_flip() +
  labs(title = "Average Feature Relevance")

# Another version of global boxplot
boxplot(lrp_result, preprocess_FUN = identity) +
  theme_bw() +
  coord_flip() +
  labs(title = "Average Feature Relevance")
