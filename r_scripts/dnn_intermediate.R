##=============================================================================##
## Title: Deep neural network for classification - intermediate using torch/luz
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: February 12, 2026
##=============================================================================##

# DNN using torch/luz
# Using dataset()
# Using callbacks

# Packages ----------------------------------------------------------------

library(torch)
library(luz)
library(tidyverse)
library(tidymodels)
library(MLDataR)

# Data --------------------------------------------------------------------

heart_df <- 
  heartdisease  %>% 
  mutate(across(c(Sex, RestingECG, Angina), as.factor))

## Explore data ----
skimr::skim(heart_df)


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

# Training plot ----------------------------------------------------------

fitted %>% plot()

# Better plot
hist <- get_metrics(fitted)

optimal_epoch <- hist %>%
  filter(metric == "loss", set == "valid") %>%
  slice_min(value, n = 1) %>%
  pull(epoch)

hist %>%
  ggplot(aes(x = epoch, y = value, color = set)) +
  geom_line(linewidth = 1) +      # Draw lines
  geom_point(size = 1.5) +        # Add points for clarity
  facet_wrap(~ metric, scales = "free_y", ncol = 1) + # Stack metrics vertically
  theme_minimal() +
  labs(
    title = "Training vs Validation Metrics",
    y = "Value",
    x = "Epoch",
    color = "Dataset"
  )


# Re-fit the model --------------------------------------------------------

# No need to refit since we use callbacks to save the best model
# see ?luz_callback_keep_best_model()

# Predict testing set ---------------------------------------------------

y_pred <- fitted %>% predict(test_dl)
y_true <- ds_tensor$y[test_ds$indices] %>% as_array()

dat_pred <- 
  y_pred %>% 
  as_array() %>% 
  as_tibble() %>% 
  rename(prob = V1) %>% 
  mutate(
    pred = ifelse(prob > 0.5, 1, 0),
    true = y_true
  ) %>% 
  mutate(across(c(pred, true), as.factor))
dat_pred

# Evaluate ---------------------------------------------------------------

fitted %>% evaluate(test_dl) # Less accurate

# Confusion matrix
dat_pred %>% 
  conf_mat(true, pred) %>% 
  autoplot("heatmap")

# Accuracy
dat_pred %>% 
  accuracy(truth = true, estimate = pred)

# Plot ROC
dat_pred %>% 
  roc_curve(true, prob, event_level = "second") %>% 
  autoplot()

dat_pred  %>% 
  roc_auc(true, prob, event_level = "second")

