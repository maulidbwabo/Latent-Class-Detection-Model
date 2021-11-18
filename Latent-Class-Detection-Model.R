##The Paper one
##Data scalling and Centering
library(ggplot2)
library(cowplot)
library(data.table)
library(viridisLite)
library(viridis)
install.packages("RSNNS")
library(Rcpp)
library(RSNNS)
install.packages("kernlab")
library(kernlab)
library(rpart)
install.packages("rattle")
install.packages("tibble")
install.packages("bitops")
library(tibble)
library(bitops)
library(rattle)
install.packages("DALEX")
library(DALEX)
library(lattice)
library(ggplot2)
library(caret)
install.packages("sp")
install.packages("spData")
install.packages("sf")
library(sp)
library(spData)
library(sf)
install.packages("spdep")
library(spdep)
install.packages("ranger")
library(ranger)
library(e1071)
install.packages("gbm")
library(gbm)
library(plyr)
set.seed(1234)
library(data.table)
dynamic=read.csv("C:/Users/bwabo/OneDrive/Desktop/Ph.D. Thesis/Data set.csv",header = TRUE,sep = ",",stringsAsFactors = FALSE)
head(dynamic)
View(dynamic)
##Principal components analysis(Exploratory Data Analysis)
dynamic = as.data.table(dynamic)
dynamicP=subset(dynamic,select = -c(1:4,6:14))
head(dynamicP)
str(dynamicP)
d = copy(dynamicP)
sort(unique(d$Region))
str(d)
#re-naming of the column
d=colnames(dynamicP)
d=gsub("[[:punct:]]", "", d)
(y = abbreviate(d, minlength = 3, method = "both.sides"))
#shorten regional names to abbreviations.
d$Region=abbreviate(d$Region,minlength = 3,method = "left.kept")
dynamicP$Region
head(d)
#Hot coding or dummy coding to the character variable(Regional)
ddum = dummyVars("~.", data = d)
d= data.table(predict(ddum, newdata = d))
str(d)
rm(ddum) #remove ddum as unneeded
#scale and centering of the data set
dscaled=scale(d[,-(1:3)])
str(d)
dscaled=as.data.table(dscaled)
d=cbind(d[,c(1:3)],dscaled)
#Transformation (Evaluating if the data have a bunches of outliers)
boxplot(d[,-c(1:3)], las = 2)
##Shapiro Test
par(mfrow = c(1,2))
hist(d$KS3, 100)
qqnorm(d$KS3)
par(mfrow = c(1,1))
shapiro.test(d$KS3)
range(d$KS3)
range(dynamicP$KS3)
shapiro.test(log(d$KS3))
##Visually transformed data
par(mfrow = c(1,2))
hist(d$KS3, 100)
hist( log (d$KS3) , 100)
par(mfrow = c(1,1))
#Observed the correlations between the transformed  and non-transformed indicators
d1 = copy(dynamicP[,.(E1, KS3)])
d1[, Log.KS3 := log(KS3)]
cor(d1)
rm(d1)
#Training and Validation of the data
set.seed(1234)
index = createDataPartition(dynamic$Region, p = 0.8, list = FALSE)
trainData = dynamic[index, ]
validationData = dynamic[-index, ]
##PCA
#confirm structure
str(trainData[,c(15:19,20:25)])
#base R / traditional method
pc = prcomp(trainData[,c(15:19,20:25)], center = TRUE, scale. = TRUE)
summary(pc)
pcValidationData1= predict(pc, newdata = validationData[,c(15:20,20:25)])
#scalable method using PCA Methods
pc=pca(trainData[,c(15:19,20:25)], method = "svd",nPcs = 4, scale = "uv", 
        center = TRUE)
pc
summary(pc)
pcValidationData2 = predict(pc, newdata = validationData[,c(15:19,20:25)])
# Accessing the transformed validation data
pcValidationData1[,1]
pcValidationData2$scores
pcValidationData2$scores[,1]
##Support Vector Machines
svmDataTrain = trainData[,.(KS3, E1)]
svmDataValidate = validationData[,.(KS3, E1)]
#Visualizing the indicators 
p1 = ggplot(data = svmDataTrain,
             aes(x = KS3, y = E1))
## data points colored by Regions
p1 + geom_point(aes(colour = trainData$Region)) +
  scale_colour_viridis(discrete = TRUE)
##SVM
set.seed(12345)
svm = train(x = svmDataTrain,
             y = trainData$Region,
             method = "svmLinear",
             preProcess = NULL,
             metric = "Accuracy",
             trControl = trainControl(method = "cv",
                                      number = 5,
                                      seeds = c(123, 234, 345, 456, 567, 678)
             )
)
svm
#predict the Region name on training data using our new model
predictOnTrain = predict(svm, newdata = svmDataTrain)
mean( predictOnTrain == trainData$Region)
#predictions 
predictOnTest =predict(svm, newdata = svmDataValidate)
mean(predictOnTest == validationData$Region)
# set up training & validation data
svmDataTrain = trainData[,-1]
svmDataTrain[,E1:=as.numeric(E1)]
svmDataValidation = validationData[,-1]
svmDataValidation[,E1:=as.numeric(E1)]
#run linear SVM on the full data set
#This is pivotal test to avoid the risk of over fitting of the model 
set.seed(12345)
svmLinear = train(x = svmDataTrain,
                   y = trainData$Region,
                   method = "svmLinear",
                   preProcess = c("scale", "center", "pca"),
                   metric = "Accuracy",
                   trControl = trainControl(method = "cv",
                                            number = 4,
                                            seeds = c(123, 234, 345, 456, 567, 678)
                   )
)
svmLinear
#Polynomial 
set.seed(12345)
svmPoly = train(x = svmDataTrain,
                 y = trainData$Region,
                 method = "svmPoly",
                 preProcess = c("scale", "center", "pca"),
                 metric = "Accuracy",
                 trControl = trainControl(method = "cv",
                                          number = 5
                 )
)
svmPoly
#remove d2 as unneeded
##CLASSIFICATION REGRESSION TREE
cartDataTrain = copy(trainData[,-1])
cartDataTrain[,KS3:=as.numeric(KS3)]
cartDataValidation = copy(validationData[,-1])
cartDataValidation[,KS3:=as.numeric(KS3)]

set.seed(12345)
cartModel <- train(x = cartDataTrain,
                   y = trainData$Region,
                   method = "rpart",
                   preProcess = c("scale", "center", "pca"),
                   metric = "Accuracy",
                   tuneLength = 10,
                   trControl = trainControl(method = "cv",
                                            number = 5
                   )
)
CartModel
rm(dscaled)
#structure of the data
str(dynamicP)
str(d)
str(d)
str(dynamicP)
#PC analysis(Transformation of data)
log.dynamicP = prcomp(dynamicP, center = TRUE, scale. = TRUE)
log.dynamicP
log.dynamicP$rotation
log.dynamicP$x
log.dynamicP$sdev
log.dynamicP$scale
dynamicRotationAbs = abs(log.dynamicP$rotation)
dynamicRotationAbs
plot(log.dynamicP, type = "l")
str(log.dynamicP)
summary(log.dynamicP)
#VQV issue
install.packages("ggbiplot")
library(ggbiplot)
install.packages("remotes")
remotes::install_github("vqv/ggbiplot")
library(ggbiplot)
require(ggplot2)
require(plyr)
require(scales)
require(grid)
ggbiplot(log.dynamicP)
ggbiplot(log.dynamicP, labels=rownames(log.dynamicP))
predict(log.dynamicP,
        newdata=tail(dynamicP, 2))
##Check for the indicators correlations 
cor(dynamicP$KS1,dynamicP$KS2)
#Graph
ggplot(dynamicP, aes(KS1, KS2)) +
  geom_point()
#PCA Package
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("pcaMethods")
biocLite("pcaMethods")
install.packages("pcaMethods")
library(Biobase)
library(BiocGenerics)
library(pcaMethods)
#list PCA Methods
listFromMethods()
##confirm the structure of the data set 
# selecting row as the training set
str(dynamicP[,c(1:48)])
#base R / traditional method
pc = prcomp(dynamicP[,c(1:48)], center = TRUE, scale. = TRUE)

summary(pc)
#PCA validation 
pcValidationData1 = predict(pc, newdata = validationData[,c(1:48)])
pcValidationD1=predict(pc,newdata=dynamicP[,c(1:48)])
pcValidationD1
#scalable method using PcaMethods
pc=pca(dynamicP[,c(1:48)], method = "svd",nPcs = 4, scale = "uv", 
        center = TRUE)
pcN=pca(dynamicP[,c(1:48)],method = "nipals",nPcs=4,scale = "uv",
        center = TRUE)
##
zu = prep(t(dynamicP))
zc=nipalsPca(zu,nPcs = 2)
summary(zc)
loadings(zc)
zc1=pca(t(dynamicP),method = "nipals", nPcs = 2)
zc1
#call the pc
pc
pcN
#summary
summary(pc)
plot(pc)
head(scores(pc))
round(cor(scores(pc)),2)
## Scores and loadings plot
loadings(pc)
loadings(pcN)
slplot(pc)
slplot(pcN)
biplot(pc, main = "Biplot of PCA")
##
pcP=pca(t(dynamicP),method = "nipals")
#predictions
pcValidationData2=predict(pc,newdata = dynamicP[,c(1:48)])
pcValidationData2$x
pcValidationData2$scores
#accessing the transformed data that has been validated
pcValidationD1[,1]
pcValidationD1[,3]
pcValidationData2$scores[,1]
pcValidationData2$scores[,2]
pcValidationData2$x[,1]
##Robust PCA
dynamicP=cbind(dynamicP[,c(11)])
str(dynamicP)
d = cbind(d[,c(1:9)], dScaled)