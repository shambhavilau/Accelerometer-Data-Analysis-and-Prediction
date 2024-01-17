# Load Libraries
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(ggplot2)

# read data from csv files
train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

head(train,n=5)
head(test,n=5)

names(train)
dim(train)
dim(test)

# Data cleaning
# Clean dataset by removing rows and columns with missing values
trainClean <- train[, colSums(is.na(train)) == 0] 
testClean <- test[, colSums(is.na(test)) == 0] 

trainClean <- trainClean[,-c(1:7)] #removing metadata which is irrelevant to the outcome
nearZeroVar(trainClean) # no variable with near zero variance

# split data into train and test from the training set
set.seed(42)
inTrain <- createDataPartition(y=trainClean$classe, p=0.6, list=FALSE)
myTrain <- trainClean[inTrain, ]
myTest <- trainClean[-inTrain, ]
dim(myTrain)
dim(myTest)

# Create Regression Models
# setting control variable for 4 fold cross validation
control <- trainControl(method="cv", number=4, verboseIter=F)

# Fitting Decision Tree Model
mod1 <- train(classe~., data=myTrain, method="rpart", trControl = control, tuneLength = 5)
rpart.plot(mod1$finalModel)
pred1 <- predict(mod1, myTest)
cm1 <- confusionMatrix(pred1, factor(myTest$classe))
cm1
plot(mod1)

# Fitting Random Forest Model
mod2 <- train(classe~., data=myTrain, method="rf", trControl = control, tuneLength = 5)
pred2 <- predict(mod2, myTest)
cm2 <- confusionMatrix(pred2, factor(myTest$classe))
cm2
plot(mod2)

# Fitting SVM Model
mod3 <- train(classe~., data=myTrain, method="svmLinear", trControl = control, tuneLength = 5, verbose = F)
pred3 <- predict(mod3, myTest)
cm3 <- confusionMatrix(pred3, factor(myTest$classe))
cm3


# Compile Model Results
ModelName <- c("Decision Tree","Random Forest","SVM")
OOSE <- c(1 - as.numeric(cm1$overall[1]),1 - as.numeric(cm2$overall[1]),1 - as.numeric(cm3$overall[1]))
Accuracy <- c(0.5442,0.9944,0.7822)
Model_Summary <- data.frame(ModelName,Accuracy,OOSE)
Model_Summary

# Prediction on Tests set
pred <- predict(mod2, testClean)
print(pred)


