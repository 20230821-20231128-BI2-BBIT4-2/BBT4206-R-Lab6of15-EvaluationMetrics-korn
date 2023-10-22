#Import the Customer_Churn dataset before running the code

# STEP 1. Install and Load the Required Packages ----

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## broom ----
if (require("broom")) {
  require("broom")
} else {
  install.packages("broom", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## modelr ----
if (require("modelr")) {
  require("modelr")
} else {
  install.packages("modelr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

####Daily_Demand_Forecasting_Orders dataset----
## Load the dataset ----
data(Daily_Demand_Forecasting_Orders)
summary(Daily_Demand_Forecasting_Orders)
Daily_Demand_Forecasting_Orders_no_na <- na.omit(Daily_Demand_Forecasting_Orders)



# RMSE, R Squared, and MAE ---
##Split the dataset ----
set.seed(7)
train_index <- sample(1:dim(Daily_Demand_Forecasting_Orders)[1], 10) # nolint: seq_linter.
Daily_Demand_Forecasting_Orders_train <- Daily_Demand_Forecasting_Orders[train_index, ]
Daily_Demand_Forecasting_Orders_test <- Daily_Demand_Forecasting_Orders[-train_index, ]

## Train the Model ----
# Apply bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)
Daily_Demand_Forecasting_Orders_model_lm <-
  train(Target..Total.orders. ~ ., data = Daily_Demand_Forecasting_Orders_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

## Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
print(Daily_Demand_Forecasting_Orders_model_lm)

### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(Daily_Demand_Forecasting_Orders_model_lm, Daily_Demand_Forecasting_Orders_test[, 1:12])
print(predictions)

#### RMSE ----
rmse <- sqrt(mean((Daily_Demand_Forecasting_Orders_test$Target..Total.orders. - predictions)^2))
print(paste("RMSE =", rmse))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((Daily_Demand_Forecasting_Orders_test$Target..Total.orders. - predictions)^2)
print(paste("SSR =", ssr))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((Daily_Demand_Forecasting_Orders_test$Target..Total.orders. - mean(Daily_Demand_Forecasting_Orders_test$Target..Total.orders.))^2)
print(paste("SST =", sst))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))

#### MAE ----
# MAE measures the average absolute differences between the predicted and
# actual values in a dataset.
absolute_errors <- abs(predictions - Daily_Demand_Forecasting_Orders_test$Target..Total.orders.)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))


####Customer_Churn dataset----
### Load the dataset ---
#data(Customer_Churn)

### Building the regression model---
model1 <- lm(Churn ~., data = Customer_Churn)
model2 <- lm(Churn ~. -Age, data = Customer_Churn)

### Assessing model quality---
#summary() returns the R-squared, adjusted R-squared and the RSE
#AIC() and BIC() compute the AIC and the BIC, respectively
summary(model1)
AIC(model1)
BIC(model1)

#computes the R2, RMSE and the MAE
data.frame(
  R2 = rsquare(model1, data = Customer_Churn),
  RMSE = rmse(model1, data = Customer_Churn),
  MAE = mae(model1, data = Customer_Churn)
)

predictions <- model1 %>% predict(Customer_Churn)
data.frame(
  R2 = R2(predictions, Customer_Churn$Churn),
  RMSE = RMSE(predictions, Customer_Churn$Churn),
  MAE = MAE(predictions, Customer_Churn$Churn)
)

#compute the R2, adjusted R2, sigma (RSE), AIC, BIC.
glance(model1)

### Manual computation of R2, MSE, RMSE and MAE---
Customer_Churn %>%
  add_predictions(model1) %>%
  summarise(
    R2 = cor(Churn, pred)^2,
    MSE = mean((Churn - pred)^2),
    RMSE = sqrt(MSE),
    MAE = mean(abs(Churn - pred))
  )

### Metrics for model 1---
glance(model1) %>%
  dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)

### Metrics for model 2---
glance(model2) %>%
  dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)

#The two models have exactly the samed adjusted R2 (0.445), meaning that they are 
#equivalent in explaining the outcome, here Churn score.
#Additionally, they have the same amount of residual standard error (RSE or sigma = 0.271).
#However, model 2 is more simple than model 1 because it incorporates less variables.
#All things equal, the simple model is always better in statistics.

#The BIC of model 2 is lower than that of the model 1.
#The model with the lowest AIC and BIC score is preferred.
