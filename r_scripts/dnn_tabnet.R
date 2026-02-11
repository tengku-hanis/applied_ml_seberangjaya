##=============================================================================##
## Title: Deep neural network for classification - basic using tidymodels
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: February 12, 2026
##=============================================================================##

# DNN using tidymodels

# Packages ----------------------------------------------------------------

library(torch)
library(tabnet)
library(tidyverse)
library(tidymodels)
library(vip) # to plot feature importances
library(MLDataR)

# Data --------------------------------------------------------------------

heart_df <- 
  heartdisease %>% 
  mutate(HeartDisease = as.factor(HeartDisease)) %>% 
  mutate(across(c(Sex, RestingECG, Angina), as.factor))

## Explore data ----
skimr::skim(heart_df)

## Split data ----
set.seed(123)
split_ind <- initial_split(heart_df, strata = "HeartDisease")
heart_train <- training(split_ind)
heart_test <- testing(split_ind)

## Preprocessing ----
heart_rc <- 
  recipe(HeartDisease ~., data = heart_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_factor_predictors())

## 10-fold CV ----
set.seed(123)
heart_cv <- vfold_cv(heart_train, v = 10)


# Fit resamples -----------------------------------------------------------

## Specify model ----
dnn_mod <- 
  mod <- tabnet(epochs = 50, batch_size = 128) %>%
  set_engine("torch") %>%
  set_mode("classification")

## Specify workflow ----
dnn_wf <- workflow() %>% 
  add_model(dnn_mod) %>% 
  add_recipe(heart_rc)      

## Fit resamples ----
set.seed(123)
dnn_res <- 
  dnn_wf %>% 
  fit_resamples(heart_cv)

dnn_res %>% 
  collect_metrics()

dnn_res %>%
  collect_metrics(summarize = FALSE) %>% # Get results for each of the 10 folds
  ggplot(aes(x = .metric, y = .estimate, fill = .metric)) +
  geom_boxplot() +
  facet_wrap(~.metric, scales = "free") +
  theme_minimal() +
  labs(title = "Metric Distribution across CV Folds")

        
# Fit to training data ----------------------------------------------------

dnn_trained <- 
  dnn_wf %>% 
  fit(heart_train)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
heart_pred <- 
  heart_test %>% 
  bind_cols(predict(dnn_trained, new_data = heart_test)) %>% 
  bind_cols(predict(dnn_trained, new_data = heart_test, type = "prob"))

## Performance metrics ----
## Accuracy
heart_pred %>% 
  accuracy(truth = HeartDisease, estimate = .pred_class)

## Plot ROC
heart_pred %>% 
  roc_curve(HeartDisease, .pred_1, event_level = "second") %>% 
  autoplot()

heart_pred %>% 
  roc_auc(HeartDisease, .pred_1, event_level = "second")


# Variable importance -----------------------------------------------------

fit <- extract_fit_parsnip(dnn_trained)
vip(fit) + theme_minimal()


