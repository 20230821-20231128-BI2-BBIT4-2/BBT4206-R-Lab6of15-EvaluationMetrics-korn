Business Intelligence Project
================
korn
23/10/23

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
  - [LAB 6](#lab-6)
  - [STEP 1. INSTALL PACKAGES](#step-1-install-packages)
  - [STEP 2. LOAD THE DATASET](#step-2-load-the-dataset)
  - [Train the Model —-](#train-the-model--)
  - [Display the different Model Performances
    —-](#display-the-different-model-performances--)
- [STEP 2.b) Customer_Churn dataset—-](#step-2b-customer_churn-dataset-)
  - [Building the regression model—](#building-the-regression-model)

# Student Details

<table>
<colgroup>
<col style="width: 53%" />
<col style="width: 46%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Numbers and Names of Group Members</strong></td>
<td><ol type="1">
<li><p>134644 - C - Sebastian Muramara</p></li>
<li><p>136675 - C - Bernard Otieno</p></li>
<li><p>131589 - C - Agnes Anyango</p></li>
<li><p>131582 - C - Njeri Njuguna</p></li>
<li><p>136009 - C- Sera Ndabari</p></li>
</ol></td>
</tr>
<tr class="even">
<td><strong>GitHub Classroom Group Name</strong></td>
<td>Korn</td>
</tr>
<tr class="odd">
<td><strong>Course Code</strong></td>
<td>BBT4206</td>
</tr>
<tr class="even">
<td><strong>Course Name</strong></td>
<td>Business Intelligence II</td>
</tr>
<tr class="odd">
<td><strong>Program</strong></td>
<td>Bachelor of Business Information Technology</td>
</tr>
<tr class="even">
<td><strong>Semester Duration</strong></td>
<td>21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023</td>
</tr>
</tbody>
</table>

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

## LAB 6

## STEP 1. INSTALL PACKAGES

``` r
# STEP 1. Install and Load the Required Packages ----

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggplot2

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: lattice

``` r
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: mlbench

``` r
## broom ----
if (require("broom")) {
  require("broom")
} else {
  install.packages("broom", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: broom

``` r
## modelr ----
if (require("modelr")) {
  require("modelr")
} else {
  install.packages("modelr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: modelr

    ## 
    ## Attaching package: 'modelr'

    ## The following object is masked from 'package:broom':
    ## 
    ##     bootstrap

``` r
## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: pROC

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

## STEP 2. LOAD THE DATASET

\####Daily_Demand_Forecasting_Orders dataset—-

``` r
##CLASSIFICATION
####BreastCancer dataset----
## Load the dataset ----
data(BreastCancer)
View(BreastCancer)
summary(BreastCancer)
```

    ##       Id             Cl.thickness   Cell.size     Cell.shape  Marg.adhesion
    ##  Length:699         1      :145   1      :384   1      :353   1      :407  
    ##  Class :character   5      :130   10     : 67   2      : 59   2      : 58  
    ##  Mode  :character   3      :108   3      : 52   10     : 58   3      : 58  
    ##                     4      : 80   2      : 45   3      : 56   10     : 55  
    ##                     10     : 69   4      : 40   4      : 44   4      : 33  
    ##                     2      : 50   5      : 30   5      : 34   8      : 25  
    ##                     (Other):117   (Other): 81   (Other): 95   (Other): 63  
    ##   Epith.c.size  Bare.nuclei   Bl.cromatin  Normal.nucleoli    Mitoses   
    ##  2      :386   1      :402   2      :166   1      :443     1      :579  
    ##  3      : 72   10     :132   3      :165   10     : 61     2      : 35  
    ##  4      : 48   2      : 30   1      :152   3      : 44     3      : 33  
    ##  1      : 47   5      : 30   7      : 73   2      : 36     10     : 14  
    ##  6      : 41   3      : 28   4      : 40   8      : 24     4      : 12  
    ##  5      : 39   (Other): 61   5      : 34   6      : 22     7      :  9  
    ##  (Other): 66   NA's   : 16   (Other): 69   (Other): 69     (Other): 17  
    ##        Class    
    ##  benign   :458  
    ##  malignant:241  
    ##                 
    ##                 
    ##                 
    ##                 
    ## 

``` r
## removing ID column since its not needed for classification
df <- subset(BreastCancer, select = -Id )
BreastCancer_no_na <- na.omit(df)
# RMSE, R Squared, and MAE ---
##Split the dataset ----
```

\##Split the dataset —-

``` r
##Split the dataset ----
BreastCancer_freq <- BreastCancer_no_na$Class
cbind(frequency =
        table( BreastCancer_freq),
      percentage = prop.table(table( BreastCancer_freq)) * 100)
```

    ##           frequency percentage
    ## benign          444   65.00732
    ## malignant       239   34.99268

``` r
train_index <- sample(1:dim(BreastCancer_no_na)[1], 10) # nolint: seq_linter.
BreastCancer_train <- BreastCancer_no_na[train_index, ]
BreastCancer_test <- BreastCancer_no_na[-train_index, ]
```

## Train the Model —-

``` r
## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

set.seed(7)
BreastCancer_model_glm <-
  train(Class ~ ., data = BreastCancer_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

print(BreastCancer_model_glm)
```

    ## Generalized Linear Model 
    ## 
    ## 10 samples
    ##  9 predictor
    ##  2 classes: 'benign', 'malignant' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 8, 8, 8, 9, 7 
    ## Resampling results:
    ## 
    ##   Accuracy  Kappa    
    ##   0.9       0.6666667

## Display the different Model Performances —-

``` r
predictions <- predict(BreastCancer_model_glm, BreastCancer_test[, 1:9])
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer_test[, 1:10]$Class)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##            Reference
    ## Prediction  benign malignant
    ##   benign       434       122
    ##   malignant      3       114
    ##                                          
    ##                Accuracy : 0.8143         
    ##                  95% CI : (0.7828, 0.843)
    ##     No Information Rate : 0.6493         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.5386         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.9931         
    ##             Specificity : 0.4831         
    ##          Pos Pred Value : 0.7806         
    ##          Neg Pred Value : 0.9744         
    ##              Prevalence : 0.6493         
    ##          Detection Rate : 0.6449         
    ##    Detection Prevalence : 0.8262         
    ##       Balanced Accuracy : 0.7381         
    ##                                          
    ##        'Positive' Class : benign         
    ## 

``` r
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")
```

![](Lab-Submission-Markdown_files/figure-gfm/model%20performance-1.png)<!-- -->

``` r
## 3.a. Load the dataset ----
data(BreastCancer)
## 3.b. Determine the Baseline Accuracy ----
# The baseline accuracy is 65%.

BreastCancer_freq <- BreastCancer_no_na$Class
cbind(frequency =
        table(BreastCancer_freq),
      percentage = prop.table(table(BreastCancer_freq)) * 100)
```

    ##           frequency percentage
    ## benign          444   65.00732
    ## malignant       239   34.99268

``` r
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
```

    ## k-Nearest Neighbors 
    ## 
    ## 513 samples
    ##   9 predictor
    ##   2 classes: 'benign', 'malignant' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (15 fold) 
    ## Summary of sample sizes: 479, 479, 478, 479, 478, 479, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  ROC        Sens       Spec     
    ##   5  0.9798419  0.9790514  0.8555556
    ##   7  0.9796333  0.9820817  0.8277778
    ##   9  0.9822958  0.9820817  0.7833333
    ## 
    ## ROC was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

``` r
predictions <- predict(BreastCancer_model_knn, BreastCancer_test[, 1:9])

print(predictions)
```

    ##   [1] benign    benign    benign    malignant benign    malignant benign   
    ##   [8] benign    malignant benign    benign    malignant malignant benign   
    ##  [15] benign    benign    benign    benign    malignant benign    malignant
    ##  [22] benign    malignant benign    malignant benign    benign    benign   
    ##  [29] benign    benign    benign    benign    malignant malignant benign   
    ##  [36] benign    benign    benign    benign    benign    malignant benign   
    ##  [43] benign    malignant malignant benign    benign    malignant benign   
    ##  [50] malignant benign    malignant malignant benign    malignant benign   
    ##  [57] benign    malignant malignant benign    malignant benign    benign   
    ##  [64] malignant benign    malignant malignant malignant benign    malignant
    ##  [71] malignant benign    benign    malignant benign    benign    malignant
    ##  [78] benign    benign    malignant benign    benign    benign    benign   
    ##  [85] benign    benign    malignant malignant benign    benign    benign   
    ##  [92] benign    benign    malignant benign    benign    benign    benign   
    ##  [99] malignant benign    malignant benign    benign    benign    benign   
    ## [106] benign    benign    benign    benign    benign    malignant benign   
    ## [113] malignant benign    benign    benign    benign    malignant benign   
    ## [120] benign    benign    benign    benign    benign    benign    benign   
    ## [127] benign    benign    benign    benign    benign    benign    benign   
    ## [134] benign    benign    benign    benign    benign    benign    malignant
    ## [141] benign    malignant benign    benign    malignant benign    benign   
    ## [148] malignant malignant benign    benign    benign    benign    benign   
    ## [155] benign    malignant benign    benign    benign    benign    benign   
    ## [162] benign    benign    benign    malignant benign    benign    benign   
    ## [169] benign    benign   
    ## Levels: benign malignant

``` r
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer_test[, 1:10]$Class)

print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##            Reference
    ## Prediction  benign malignant
    ##   benign       111        13
    ##   malignant      0        46
    ##                                           
    ##                Accuracy : 0.9235          
    ##                  95% CI : (0.8728, 0.9587)
    ##     No Information Rate : 0.6529          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8221          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0008741       
    ##                                           
    ##             Sensitivity : 1.0000          
    ##             Specificity : 0.7797          
    ##          Pos Pred Value : 0.8952          
    ##          Neg Pred Value : 1.0000          
    ##              Prevalence : 0.6529          
    ##          Detection Rate : 0.6529          
    ##    Detection Prevalence : 0.7294          
    ##       Balanced Accuracy : 0.8898          
    ##                                           
    ##        'Positive' Class : benign          
    ## 

``` r
#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(BreastCancer_model_knn, BreastCancer_test[, 1:9],
                       type = "prob")

# These are the class probability values for diabetes that the
# model has predicted:
print(predictions)
```

    ##         benign  malignant
    ## 1   0.69230769 0.30769231
    ## 2   1.00000000 0.00000000
    ## 3   1.00000000 0.00000000
    ## 4   0.00000000 1.00000000
    ## 5   0.95454545 0.04545455
    ## 6   0.11111111 0.88888889
    ## 7   0.79166667 0.20833333
    ## 8   0.52631579 0.47368421
    ## 9   0.00000000 1.00000000
    ## 10  0.62962963 0.37037037
    ## 11  1.00000000 0.00000000
    ## 12  0.09523810 0.90476190
    ## 13  0.44444444 0.55555556
    ## 14  1.00000000 0.00000000
    ## 15  1.00000000 0.00000000
    ## 16  1.00000000 0.00000000
    ## 17  0.71428571 0.28571429
    ## 18  0.50000000 0.50000000
    ## 19  0.00000000 1.00000000
    ## 20  1.00000000 0.00000000
    ## 21  0.00000000 1.00000000
    ## 22  1.00000000 0.00000000
    ## 23  0.22222222 0.77777778
    ## 24  1.00000000 0.00000000
    ## 25  0.00000000 1.00000000
    ## 26  0.57894737 0.42105263
    ## 27  1.00000000 0.00000000
    ## 28  1.00000000 0.00000000
    ## 29  1.00000000 0.00000000
    ## 30  1.00000000 0.00000000
    ## 31  1.00000000 0.00000000
    ## 32  1.00000000 0.00000000
    ## 33  0.26666667 0.73333333
    ## 34  0.30769231 0.69230769
    ## 35  1.00000000 0.00000000
    ## 36  1.00000000 0.00000000
    ## 37  1.00000000 0.00000000
    ## 38  1.00000000 0.00000000
    ## 39  1.00000000 0.00000000
    ## 40  1.00000000 0.00000000
    ## 41  0.00000000 1.00000000
    ## 42  1.00000000 0.00000000
    ## 43  1.00000000 0.00000000
    ## 44  0.00000000 1.00000000
    ## 45  0.00000000 1.00000000
    ## 46  1.00000000 0.00000000
    ## 47  1.00000000 0.00000000
    ## 48  0.00000000 1.00000000
    ## 49  1.00000000 0.00000000
    ## 50  0.20000000 0.80000000
    ## 51  1.00000000 0.00000000
    ## 52  0.00000000 1.00000000
    ## 53  0.00000000 1.00000000
    ## 54  1.00000000 0.00000000
    ## 55  0.00000000 1.00000000
    ## 56  1.00000000 0.00000000
    ## 57  0.97058824 0.02941176
    ## 58  0.00000000 1.00000000
    ## 59  0.22222222 0.77777778
    ## 60  1.00000000 0.00000000
    ## 61  0.43750000 0.56250000
    ## 62  0.68181818 0.31818182
    ## 63  1.00000000 0.00000000
    ## 64  0.28571429 0.71428571
    ## 65  1.00000000 0.00000000
    ## 66  0.09090909 0.90909091
    ## 67  0.22222222 0.77777778
    ## 68  0.00000000 1.00000000
    ## 69  1.00000000 0.00000000
    ## 70  0.08333333 0.91666667
    ## 71  0.31578947 0.68421053
    ## 72  0.61538462 0.38461538
    ## 73  0.95555556 0.04444444
    ## 74  0.00000000 1.00000000
    ## 75  1.00000000 0.00000000
    ## 76  1.00000000 0.00000000
    ## 77  0.17647059 0.82352941
    ## 78  1.00000000 0.00000000
    ## 79  1.00000000 0.00000000
    ## 80  0.09090909 0.90909091
    ## 81  0.90909091 0.09090909
    ## 82  1.00000000 0.00000000
    ## 83  1.00000000 0.00000000
    ## 84  0.64705882 0.35294118
    ## 85  1.00000000 0.00000000
    ## 86  1.00000000 0.00000000
    ## 87  0.18181818 0.81818182
    ## 88  0.00000000 1.00000000
    ## 89  0.90000000 0.10000000
    ## 90  1.00000000 0.00000000
    ## 91  1.00000000 0.00000000
    ## 92  1.00000000 0.00000000
    ## 93  0.75000000 0.25000000
    ## 94  0.00000000 1.00000000
    ## 95  1.00000000 0.00000000
    ## 96  1.00000000 0.00000000
    ## 97  1.00000000 0.00000000
    ## 98  1.00000000 0.00000000
    ## 99  0.00000000 1.00000000
    ## 100 1.00000000 0.00000000
    ## 101 0.12500000 0.87500000
    ## 102 1.00000000 0.00000000
    ## 103 1.00000000 0.00000000
    ## 104 1.00000000 0.00000000
    ## 105 1.00000000 0.00000000
    ## 106 1.00000000 0.00000000
    ## 107 1.00000000 0.00000000
    ## 108 0.65000000 0.35000000
    ## 109 1.00000000 0.00000000
    ## 110 1.00000000 0.00000000
    ## 111 0.00000000 1.00000000
    ## 112 1.00000000 0.00000000
    ## 113 0.00000000 1.00000000
    ## 114 1.00000000 0.00000000
    ## 115 1.00000000 0.00000000
    ## 116 1.00000000 0.00000000
    ## 117 1.00000000 0.00000000
    ## 118 0.00000000 1.00000000
    ## 119 1.00000000 0.00000000
    ## 120 1.00000000 0.00000000
    ## 121 1.00000000 0.00000000
    ## 122 1.00000000 0.00000000
    ## 123 1.00000000 0.00000000
    ## 124 1.00000000 0.00000000
    ## 125 1.00000000 0.00000000
    ## 126 1.00000000 0.00000000
    ## 127 1.00000000 0.00000000
    ## 128 1.00000000 0.00000000
    ## 129 1.00000000 0.00000000
    ## 130 1.00000000 0.00000000
    ## 131 1.00000000 0.00000000
    ## 132 1.00000000 0.00000000
    ## 133 1.00000000 0.00000000
    ## 134 1.00000000 0.00000000
    ## 135 1.00000000 0.00000000
    ## 136 1.00000000 0.00000000
    ## 137 1.00000000 0.00000000
    ## 138 1.00000000 0.00000000
    ## 139 1.00000000 0.00000000
    ## 140 0.00000000 1.00000000
    ## 141 1.00000000 0.00000000
    ## 142 0.00000000 1.00000000
    ## 143 1.00000000 0.00000000
    ## 144 1.00000000 0.00000000
    ## 145 0.10526316 0.89473684
    ## 146 1.00000000 0.00000000
    ## 147 1.00000000 0.00000000
    ## 148 0.00000000 1.00000000
    ## 149 0.00000000 1.00000000
    ## 150 1.00000000 0.00000000
    ## 151 1.00000000 0.00000000
    ## 152 1.00000000 0.00000000
    ## 153 1.00000000 0.00000000
    ## 154 1.00000000 0.00000000
    ## 155 1.00000000 0.00000000
    ## 156 0.00000000 1.00000000
    ## 157 1.00000000 0.00000000
    ## 158 1.00000000 0.00000000
    ## 159 1.00000000 0.00000000
    ## 160 0.97058824 0.02941176
    ## 161 1.00000000 0.00000000
    ## 162 1.00000000 0.00000000
    ## 163 1.00000000 0.00000000
    ## 164 0.93750000 0.06250000
    ## 165 0.28000000 0.72000000
    ## 166 1.00000000 0.00000000
    ## 167 1.00000000 0.00000000
    ## 168 1.00000000 0.00000000
    ## 169 1.00000000 0.00000000
    ## 170 1.00000000 0.00000000

``` r
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
```

    ## Setting levels: control = benign, case = malignant

    ## Setting direction: controls < cases

``` r
# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)
```

![](Lab-Submission-Markdown_files/figure-gfm/model%20performance-2.png)<!-- -->

``` r
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
```

    ## CART 
    ## 
    ## 683 samples
    ##   9 predictor
    ##   2 classes: 'benign', 'malignant' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 547, 546, 547, 546, 546, 546, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          logLoss  
    ##   0.02092050  0.2347604
    ##   0.05439331  0.2636713
    ##   0.79079498  0.3938379
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.0209205.

# STEP 2.b) Customer_Churn dataset—-

``` r
library(readr)
Customer_Churn <- read_csv("../data/Customer_Churn.csv")
```

    ## Rows: 3150 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (14): Call  Failure, Complains, Subscription  Length, Charge  Amount, Se...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(Customer_Churn)
```

## Building the regression model—

``` r
### Building the regression model---
model1 <- lm(Churn ~., data = Customer_Churn)
model2 <- lm(Churn ~. -Age, data = Customer_Churn)

### Assessing model quality---
#summary() returns the R-squared, adjusted R-squared and the RSE
#AIC() and BIC() compute the AIC and the BIC, respectively
summary(model1)
```

    ## 
    ## Call:
    ## lm(formula = Churn ~ ., data = Customer_Churn)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.92075 -0.09224 -0.02488  0.03260  0.96439 
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)               -6.532e-03  4.314e-02  -0.151 0.879637    
    ## `Call  Failure`            6.807e-03  1.138e-03   5.983 2.43e-09 ***
    ## Complains                  5.664e-01  1.955e-02  28.964  < 2e-16 ***
    ## `Subscription  Length`    -2.704e-03  6.128e-04  -4.412 1.06e-05 ***
    ## `Charge  Amount`          -2.109e-02  5.674e-03  -3.717 0.000205 ***
    ## `Seconds of Use`           1.329e-05  5.494e-06   2.418 0.015642 *  
    ## `Frequency of use`        -1.795e-03  3.657e-04  -4.907 9.71e-07 ***
    ## `Frequency of SMS`        -3.917e-04  2.839e-04  -1.380 0.167801    
    ## `Distinct Called Numbers` -1.463e-03  4.415e-04  -3.314 0.000931 ***
    ## `Age Group`               -3.365e-02  2.038e-02  -1.651 0.098852 .  
    ## `Tariff Plan`             -2.206e-02  2.278e-02  -0.968 0.333020    
    ## Status                     2.416e-01  1.521e-02  15.885  < 2e-16 ***
    ## Age                        3.109e-03  2.109e-03   1.474 0.140559    
    ## `Customer Value`           5.407e-05  6.980e-05   0.775 0.438633    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.2712 on 3136 degrees of freedom
    ## Multiple R-squared:  0.4473, Adjusted R-squared:  0.445 
    ## F-statistic: 195.2 on 13 and 3136 DF,  p-value: < 2.2e-16

``` r
AIC(model1)
```

    ## [1] 733.6785

``` r
BIC(model1)
```

    ## [1] 824.5058

``` r
#computes the R2, RMSE and the MAE
data.frame(
  R2 = rsquare(model1, data = Customer_Churn),
  RMSE = rmse(model1, data = Customer_Churn),
  MAE = mae(model1, data = Customer_Churn)
)
```

    ##          R2      RMSE       MAE
    ## 1 0.4472941 0.2705648 0.1650399

``` r
predictions <- model1 %>% predict(Customer_Churn)
data.frame(
  R2 = R2(predictions, Customer_Churn$Churn),
  RMSE = RMSE(predictions, Customer_Churn$Churn),
  MAE = MAE(predictions, Customer_Churn$Churn)
)
```

    ##          R2      RMSE       MAE
    ## 1 0.4472941 0.2705648 0.1650399

``` r
#compute the R2, adjusted R2, sigma (RSE), AIC, BIC.
glance(model1)
```

    ## # A tibble: 1 × 12
    ##   r.squared adj.r.squared sigma statistic p.value    df logLik   AIC   BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>   <dbl> <dbl>  <dbl> <dbl> <dbl>
    ## 1     0.447         0.445 0.271      195.       0    13  -352.  734.  825.
    ## # ℹ 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

``` r
### Manual computation of R2, MSE, RMSE and MAE---
Customer_Churn %>%
  add_predictions(model1) %>%
  summarise(
    R2 = cor(Churn, pred)^2,
    MSE = mean((Churn - pred)^2),
    RMSE = sqrt(MSE),
    MAE = mean(abs(Churn - pred))
  )
```

    ## # A tibble: 1 × 4
    ##      R2    MSE  RMSE   MAE
    ##   <dbl>  <dbl> <dbl> <dbl>
    ## 1 0.447 0.0732 0.271 0.165

``` r
### Metrics for model 1---
glance(model1) %>%
  dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)
```

    ## # A tibble: 1 × 5
    ##   adj.r.squared sigma   AIC   BIC p.value
    ##           <dbl> <dbl> <dbl> <dbl>   <dbl>
    ## 1         0.445 0.271  734.  825.       0

``` r
### Metrics for model 2---
glance(model2) %>%
  dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)
```

    ## # A tibble: 1 × 5
    ##   adj.r.squared sigma   AIC   BIC p.value
    ##           <dbl> <dbl> <dbl> <dbl>   <dbl>
    ## 1         0.445 0.271  734.  819.       0
