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
#CLASSIFICATION
####BreastCancer dataset----
## Load the dataset ----
data(BreastCancer)
View(BreastCancer)
summary(BreastCancer)
## removing ID column since its not needed for classification
df <- subset(BreastCancer, select = -Id )
#remove missing data
BreastCancer_no_na <- na.omit(df)
# RMSE, R Squared, and MAE ---
##Split the dataset ----
BreastCancer_freq <- BreastCancer_no_na$Class
cbind(frequency =
        table( BreastCancer_freq),
      percentage = prop.table(table( BreastCancer_freq)) * 100)

train_index <- sample(1:dim(BreastCancer_no_na)[1], 10) # nolint: seq_linter.
BreastCancer_train <- BreastCancer_no_na[train_index, ]
BreastCancer_test <- BreastCancer_no_na[-train_index, ]

## Train the Model ----
# ApplClass bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)
BreastCancer_model_lm <-
  train(Class ~ ., data = BreastCancer_train,
        na.action = na.omit, method = "glm", metric = "Accuracy",
        trControl = train_control)

BreastCancer_freq <- BreastCancer_no_na$Class
cbind(frequency =
        table(BreastCancer_freq),
      percentage = prop.table(table(BreastCancer_freq)) * 100)
## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(BreastCancer_no_na$Class,
                                   p = 0.75,
                                   list = FALSE)
BreastCancer_train <- BreastCancer_no_na[train_index, ]
BreastCancer_test <- BreastCancer_no_na[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

set.seed(7)
BreastCancer_model_glm <-
  train(Class ~ ., data = BreastCancer_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

print(BreastCancer_model_glm)


predictions <- predict(BreastCancer_model_glm, BreastCancer_test[, 1:9])
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer_test[, 1:10]$Class)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3.a. Load the dataset ----
data(BreastCancer)
## 3.b. Determine the Baseline Accuracy ----
# The baseline accuracy is 65%.

BreastCancer_freq <- BreastCancer_no_na$Class
cbind(frequency =
        table(BreastCancer_freq),
      percentage = prop.table(table(BreastCancer_freq)) * 100)

## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(BreastCancer_no_na$Class,
                                   p = 0.75,
                                   list = FALSE)
BreastCancer_train <- BreastCancer_no_na[train_index, ]
BreastCancer_test <- BreastCancer_no_na[-train_index, ]


## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 15,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

# We then train a k Nearest Neighbours Model to predict the value of Diabetes
# (whether the patient will test positive/negative for diabetes).

set.seed(7)
BreastCancer_model_knn <-
  train(Class ~ ., data =  BreastCancer_train, method = "knn",
        metric = "ROC", trControl = train_control)

print(BreastCancer_model_knn)


predictions <- predict(BreastCancer_model_knn, BreastCancer_test[, 1:9])

print(predictions)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer_test[, 1:10]$Class)

print(confusion_matrix)


#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(BreastCancer_model_knn, BreastCancer_test[, 1:9],
                       type = "prob")

# These are the class probability values for diabetes that the
# model has predicted:
print(predictions)

# "Controls" and "Cases": In a binary classification problem, you typically
# have two classes, often referred to as "controls" and "cases."
# These classes represent the different outcomes you are trying to predict.
# For example, in a medical context, "controls" might represent patients without
# a disease, and "cases" might represent patients with the disease.

# Setting the Direction: The phrase "Setting direction: controls < cases"
# specifies how you define which class is considered the positive class (cases)
# and which is considered the negative class (controls) when calculating
# sensitivity and specificity.
roc_curve <- roc(BreastCancer_test$Class, predictions$malignant)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)


train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)
# This creates a CART model. One of the parameters used by a CART model is "cp".
# "cp" refers to the "complexity parameter". It is used to impose a penalty to
# the tree for having too many splits. The default value is 0.01.
BreastCancer_model_cart <- train(Class ~ ., data = BreastCancer_no_na, method = "rpart",
                         metric = "logLoss", trControl = train_control)

## 4.c. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show that a cp value of ≈ 0 resulted in the lowest
# LogLoss value. The lowest logLoss value is ≈ 0.46.
print(BreastCancer_model_cart)


### REGRESSION

####Customer_Churn dataset----
### Load the dataset ---
library(readr)
Customer_Churn <- read_csv("data/Customer_Churn.csv")
View(Customer_Churn)
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

