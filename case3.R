# Group 4: George Garcia, Hillary Balli, Marco Perez
# Case Study #3: Implement data analysis strategies on 'BreastCancer' data set.

#####################
# Libraries and Data

library(mlbench)
library(skimr)
library(caret)
library(corrplot)
library(arm)
library(MASS)
library(pls)
library(DMwR)
library(stepPlr)
library(sda)
library(pamr)
library(rpart)
library(plyr)
library(kernlab)
library(klaR)
library(C50)
library(xgboost)
library(caretEnsemble)
data(BreastCancer)
dataBC <- na.omit(BreastCancer)
View(dataBC)

####################
# 1 | Data Splitting

# Separate the predictors (x) and outcome (y) variables
x <- subset(dataBC, select = -Class)
y <- subset(dataBC, select = Class)

# Plot distribution of the two-class outcome variable 'Class' to check for imbalance
barplot(table(y), 
        names.arg = c("Benign", "Malignant"),
        col = c("blue", "green"),
        main = "Class Distribution")

# Since the outcome data is imbalanced, use stratified random sampling and subsampling
reseed <- function() {
  set.seed(123)
}

reseed()
trainPart <- createDataPartition(y$Class, p = 0.7, list = FALSE)
xTrain <- x[trainPart,]
yTrain <- y[trainPart,]
xTest <- x[-trainPart,]
yTest <- y[-trainPart,]


#####################
# 2 | Data Summary

xSkim <- skim_to_wide(xTrain)
View(xSkim)
ySkim <- skim_to_wide(yTrain)
View(ySkim)
# There are 479 observations of 10 variables with no quantifiable skewness.
# All dependent variables are factors except 'Id'.

# Since 'Id' is not a predictive factor, omit it
xTrain <- xTrain[,-1]


#####################
# 3 | Data Visualization

# Boxplots of predictors
boxplot(xTrain)

# Set predictors as numeric
mark <- sapply(xTrain, is.factor)
xTrainNum <- as.data.frame(lapply(xTrain, function(x) as.numeric(as.character(x))))

# Histograms of predictors and outcome variables
par(mfrow = c(3,3))
hist(xTrainNum$Cl.thickness)
hist(xTrainNum$Cell.size)
hist(xTrainNum$Cell.shape)
hist(xTrainNum$Marg.adhesion)
hist(xTrainNum$Epith.c.size)
hist(xTrainNum$Bare.nuclei)
hist(xTrainNum$Bl.cromatin)
hist(xTrainNum$Normal.nucleoli)
hist(as.numeric(yTrain))

# Correlations
correlations <- cor(xTrainNum)
par(mfrow = c(1,1))
corrplot(correlations, method = "number", order = "hclust")


#######################
# 4| Omit Highly Correlated data

# Find and omit high correlations above threshold = 0.8
highCorr <- findCorrelation(correlations, cutoff = 0.8)
xTrainNum <- xTrainNum[, -highCorr]


######################
# 5 | 6 | Algorithms (with standardized cleaning)

# Controlled resampling with repeated training/test splits and smote subsampling
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     summaryFunction = twoClassSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = TRUE)

# We will treat factor variables with non-formula method, to prevent dummy variables.

# Logistic regression
reseed()
modelLR <- train(x = xTrainNum, y = yTrain, 
                 method = "bayesglm", # No tuning parameters
                 trControl = ctrl,
                 preProc = c("center", "scale"))
modelLR # ROC = 0.9931, Sens = 0.9704, Spec = 0.9533


# Linear Discriminant Analysis
reseed()
modelLDA <- train(x = xTrainNum, y = yTrain, 
                 method = "lda", # No tuning parameters
                 trControl = ctrl,
                 preProc = c("center", "scale"))
modelLDA # ROC = 0.9949, Sens = 0.9823, Spec = 0.9067


# Partial Least Squares Discriminant Analysis
gridPLS = expand.grid(ncomp = 1:5)
reseed()
modelPLS <- train(x = xTrainNum, y = yTrain, 
                  method = "pls",
                  tuneGrid = gridPLS,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
modelPLS # ROC = 0.9952, Sens = 0.9823, Spec = 0.920


# Penalized Logistic Regression
gridPLR = expand.grid(lambda = seq(0.015, 0.030, by = 0.001),
                      cp = 'bic')
reseed()
modelPLR <- train(x = xTrainNum, y = yTrain, 
                  method = "plr",
                  tuneGrid = gridPLR,
                  preProcess = c("center", "scale"),
                  trControl = ctrl)
modelPLR # ROC = 0.9942, Sens = 0.968, Spec = 0.942


# Penalized Linear Discriminant Analysis
gridPLDA = expand.grid(diagonal = c(TRUE, FALSE), lambda = seq(0.015, 0.030, by = 0.001))
reseed()
modelPLDA <- train(x = xTrainNum, y = yTrain, 
                  method = "sda",
                  tuneGrid = gridPLDA,
                  preProcess = c("center", "scale"),
                  trControl = ctrl)
modelPLDA # ROC = 0.996, Sens = 0.978, Spec = 0.933


# Nearest Shrunken Centroids
gridNSC <- data.frame(threshold = seq(0, 0.5, by = 0.01))
reseed()
modelNSC <- train(x = xTrainNum, y = yTrain, 
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = gridNSC,
                  trControl = ctrl)
modelNSC # ROC = 0.995, Sens = 0.979, Spec = 0.918


# Nonlinear Discriminant Analysis
gridMDA <- expand.grid(subclasses = 1:2)
reseed()
modelMDA <- train(x = xTrainNum, y = yTrain, 
                  method = "mda",
                  preProc = c("center", "scale"),
                  tuneGrid = gridMDA,
                  trControl = ctrl)
modelMDA # ROC = 0.995, Sens = 0.982, Spec = 0.907


# Support Vector Machines
gridSVM <- expand.grid(sigma = seq(0.001, 0.01, by = 0.001), C = c(1, 2))
reseed()
modelSVM <- train(x = xTrainNum, 
                  y = yTrain,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneGrid = gridSVM,
                  fit = FALSE,
                  trControl = ctrl)
modelSVM # ROC = 0.996, Sens = 0.972, Sped = 0.954


# K-Nearest Neighbors
gridKNN <- expand.grid(k = seq(19, 29, by = 2))
reseed()
modelKNN <- train(x = xTrainNum, 
                  y = yTrain,
                  method = "knn",
                  preProc = c("center", "scale"),
                  tuneGrid = gridKNN,
                  trControl = ctrl)
modelKNN # ROC = 0.989, Sens = 0.975, Spec = 0.951


# Naive Bayes
gridNB <- expand.grid(fL = c(0,1), usekernel = c(TRUE,FALSE), adjust = c(0,1))
reseed()
modelNB <- train( x = xTrainNum, 
                  y = yTrain,
                  method = "nb",
                  tuneGrid = gridNB,
                  preProc = c("center", "scale"),
                  trControl = ctrl)
modelNB # ROC = 0.991, Sens = 0.953, Spec = 0.977


# CART Classification trees (by rpart)
gridCART <- expand.grid(cp = c(0, 0.05, 0.1))
reseed()
modelCART <- train(x = xTrainNum, y = yTrain, 
                  method = "rpart",  
                  tuneGrid = gridCART, 
                  preProc = c("center", "scale"), 
                  trControl = ctrl)
modelCART # ROC = 0.955, Sens = 0.950, Spec = 0.910


# C4.5 Rule-based Model (by J48)
gridRBM <- expand.grid(C = 0.255, M = seq(3, 5, by = 1))
reseed()
modelRBM <- train(x = xTrainNum, y = yTrain, 
                   method = "J48",  
                   tuneGrid = gridRBM, 
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelRBM # ROC = 0.959, Sens = 0.959, Spec = 0.914


# Bagged Trees
reseed()
modelBT <- train(x = xTrainNum, y = yTrain, 
                  method = "treebag",  # No hyperparameters for this model
                  preProc = c("center", "scale"), 
                  trControl = ctrl)
modelBT # ROC = 0.983, Sens = 0.962, Spec = 0.943


# Random Forest
gridRF <- expand.grid(mtry = seq(0.5, 2, by = 0.5))
reseed()
modelRF <- train(x = xTrainNum, y = yTrain, 
                  method = "rf",  
                  tuneGrid = gridRF,
                  ntree = 1000,
                  preProc = c("center", "scale"), 
                  trControl = ctrl)
modelRF # ROC = 0.993, Sens = 0.973, Spec = 0.979


# Gradient Boosting Machine
gridGBM <- expand.grid(interaction.depth = c(9, 11, 13),
                       n.trees = (10:20),
                       shrinkage = 0.1,
                       n.minobsinnode = 10)
reseed()
modelGBM <- train(x = xTrainNum, 
                y = yTrain,
                method = "gbm",
                preProc = c("center", "scale"),
                tuneGrid = gridGBM,
                verbose = FALSE,
                trControl = ctrl)
modelGBM # ROC = 0.989, Sens = 0.971, Spec = 0.947


# C5.0
gridC5 <- expand.grid(trials = c(25:30), 
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))
reseed()
modelC5 <- train(x = xTrainNum,
                y = yTrain,
                method = "C5.0",
                preProc=c("center", "scale"),
                tuneGrid = gridC5,
                verbose = FALSE,
                trControl = ctrl)
modelC5 # ROC = 0.992, Sens = 0.965, Spec = 0.950


# Boosted C5.0
reseed()
modelBC5 <- C5.0(x = xTrainNum,
                 y = yTrain,
                 method = "C5.0",
                 trials = 50,
                 preProc=c("center", "scale"),
                 tuneGrid = gridC5,
                 verbose = FALSE,
                 trControl = ctrl)
modelBC5 # Trial 50 correctly classified 100% of the cases with 0/479 errors and may be the most overfit.


# eXtreme Gradient Boosted Tree
gridXGBT <- expand.grid(nrounds = c(50, 100), 
                      max_depth = c(1, 2),
                      eta = c(0.4, 0.5),
                      gamma = 0,
                      colsample_bytree = c(0.6, 0.8),
                      min_child_weight = 1,
                      subsample = c(0.5, 0.75, 1.0))
reseed()
modelXGBT <- train(x = xTrainNum, y = yTrain, 
                 method = "xgbTree",
                 tuneGrid = gridXGBT,
                 preProc = c("center", "scale"), 
                 trControl = ctrl)
modelXGBT # ROC = 0.991, Sens = 0.973, Spec = 0.938


# eXtreme Gradient Boosted DART
gridXGBD <- expand.grid(nrounds = c(50, 100), 
                        max_depth = c(1, 2),
                        eta = c(0.2, 0.3),
                        gamma = 0,
                        subsample = c(0.5, 0.75, 1.0),
                        colsample_bytree = c(0.6, 0.8),
                        rate_drop = c(0.01, 0.25),
                        skip_drop =  c(0.05, 0.25),
                        min_child_weight = 1)
reseed()
modelXGBD <- train(x = xTrainNum, y = yTrain, 
                   method = "xgbDART",
                   tuneGrid = gridXGBD,
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBD # ROC = 0.992, Sens = 0.972, Spec = 0.950


# eXtreme Gradient Boosted Linear
gridXGBL <- expand.grid(nrounds = c(40, 50, 60), 
                        lambda = 0,
                        alpha = 0,
                        eta = c(0.3, 0.4))
reseed()
modelXGBL <- train(x = xTrainNum, y = yTrain, 
                   method = "xgbLinear",
                   tuneGrid = gridXGBL,
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBL # ROC = 0.989, Sens = 0.967, Spec = 0.945


# Catboost


##########################
# 7 | Statistical Analysis

# Resampled model results
modelResults <- resamples(list(LR = modelLR, LDA = modelLDA, PLS = modelPLS,
                                    PLR = modelPLR, PLDA = modelPLDA, NSC = modelNSC,
                                    MDA = modelMDA, SVM = modelSVM, KNN = modelKNN,
                                    NB = modelNB, CART = modelCART, RBM = modelRBM,
                                    BT = modelBT, RF = modelRF, GBM = modelGBM, C5 = modelC5,
                                    BC5 = modelBC5, XGBT = modelXGBT, XGBD = modelXGBD, XGBL = modelXGBL))
summary(modelResults)
dotplot(modelResults)

# Check differences
modelDiff <- diff(modelResults)


########################
# 8 | Accuracy: Training Data

# In terms of accuracy, Boosted C5.0 is the best fit model.  It's boosted trial #50 correctly
# identifies 311 cases of class 'benign' and 168 cases of class 'malignant'.  A quick look at
# the class distribution for the trainig yield proves this model quite accurate.

barplot(table(yTrain), 
        names.arg = c("Benign", "Malignant"),
        col = c("blue", "green"),
        main = "Class Distribution")

########################
# 9 | Accuracy: Testing Data

# Prepare training data
preProcParams <- preProcess(xTrainNum, method = c("center", "scale"))
xTrainTrans <- predict(preProcParams, xTrainNum)

# Train final model
finalModel <-  C5.0(x = xTrainTrans, y = yTrain, trials = 50)
finalModel
summary(finalModel)

# Prepare testing data
xTestNum <- as.data.frame(lapply(xTest, function(x) as.numeric(as.character(x))))
reseed()
xTestTrans <- predict(preProcParams, xTestNum)

# Test data predictions
predictions <- predict(finalModel, newdata = xTestTrans, neighbors = 3)

# Determine accuracy
confusionMatrix(predictions, yTest)

# Accuracy on testing data is 0.985, with a sensitivity = 0.985 and specificity = 0.986.  With

########################
# 10 | caretEnsemble

# Train a list of models using caretList()
algoList <- c('C5.0', 'svmRadial', 'sda')
models <- caretList(x = xTrainNum, y = yTrain,
                    methodList = algoList,
                    tuneList = list(
                      rf1 = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry=2)),
                      rf2 = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry=10), preProcess = "pca"),
                      nn = caretModelSpec(method = "nnet", trace = FALSE, tuneLength = 2)
                    ),
                    preProc = c("center", "scale"),
                    trControl = ctrl)

# Combine with caretEnsemble()
ensemble1 <- caretEnsemble(models, metric = "Accuracy", trControl = trainControl(number = 6))
summary(ensemble1)

# caretEnsemble resulted in a combined accuracy of 0.970.

# Combine outputs with caretStack()
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions=TRUE)
reseed()
stackGLM <- caretStack(models, method="glm",  
                        preProc=c("center", "scale"), 
                        trControl=stackControl)
stackGLM # Accuracy = 0.970


reseed()
stackRF <- caretStack(models, method="rf",
                       preProc=c("center", "scale"), 
                       trControl=stackControl)
stackRF # Accuracy = 0.983