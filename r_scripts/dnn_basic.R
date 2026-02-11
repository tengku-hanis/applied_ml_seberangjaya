##=============================================================================##
## Title: Deep neural network for classification - basic using torch/luz
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: February 12, 2026
##=============================================================================##

# DNN using torch/luz
# Use dropout

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

## Split data ----
set.seed(123)
split_ind <- initial_validation_split(heart_df, strata = "HeartDisease")
heart_train <- training(split_ind)
heart_val <- validation(split_ind)
heart_test <- testing(split_ind)

## Preprocessing ----
heart_rc <- 
  recipe(HeartDisease ~., data = heart_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_factor_predictors())

heart_train_processed <- 
  heart_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

heart_val_processed <- 
  heart_rc %>% 
  prep() %>% 
  bake(new_data = heart_val)

heart_test_processed <- 
  heart_rc %>% 
  prep() %>% 
  bake(new_data = heart_test)

# Convert to dataloader --------------------------------------------------

# Convert to torch dataset
dat_train_torch <- 
  tensor_dataset(
    heart_train_processed %>% select(-HeartDisease) %>% as.matrix() %>% torch_tensor(dtype = torch_float()),
    heart_train_processed$HeartDisease %>% torch_tensor(dtype = torch_float()) %>% torch_unsqueeze(2)
  )

dat_val_torch <- 
  tensor_dataset(
    heart_val_processed %>% select(-HeartDisease) %>% as.matrix() %>% torch_tensor(dtype = torch_float()),
    heart_val_processed$HeartDisease %>% torch_tensor(dtype = torch_float()) %>% torch_unsqueeze(2)
  )

dat_test_torch <- 
  tensor_dataset(
    heart_test_processed %>% select(-HeartDisease) %>% as.matrix() %>% torch_tensor(dtype = torch_float()),
    heart_test_processed$HeartDisease %>% torch_tensor(dtype = torch_float()) %>% torch_unsqueeze(2)
  )

# Dataloader
train_dl <- dataloader(dat_train_torch, batch_size = 10, shuffle = TRUE)
val_dl <- dataloader(dat_val_torch, batch_size = 10, shuffle = FALSE)
test_dl <- dataloader(dat_test_torch, batch_size = 10, shuffle = FALSE)


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
      self$net(x)
    }
  )


# Fit the model ----------------------------------------------------------

# Set parameters
d_in <- length(heart_train_processed) - 1 # no of features minus the outcome

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
    valid_data = val_dl
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

# Fit
fitted2 <- 
  net %>% 
  setup(
    loss = nn_bce_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_binary_accuracy(threshold = 0.5), 
      luz_metric_binary_auroc(from_logits = FALSE)
    )
  ) %>% 
  set_hparams(d_in = d_in) %>% 
  fit(
    train_dl, 
    epoch = 5, 
    valid_data = val_dl
)

# Predict testing set ---------------------------------------------------

y_pred <- fitted2 %>% predict(test_dl)

dat_pred <- 
  y_pred %>% 
  as_array() %>% 
  as_tibble() %>% 
  rename(prob = V1) %>% 
  mutate(
    pred = ifelse(prob > 0.5, 1, 0),
    true = heart_test$HeartDisease
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


