# Random Forest Algorithm
install.packages('caTools')
library(caTools)


# Import the Dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the dependent variable (Needed in some classifications)
dataset$Purchased = factor(dataset$Purchased , levels = c(0,1))

# Split the data into train and test sets
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Random Forest Model
install.packages('randomForest')
library(randomForest)
model = randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 10)


# Predict the test set results
Y_pred = predict(model, newdata = test_set[-3])


# Making the confusion matrix
cm = table(test_set[ ,3], Y_pred)