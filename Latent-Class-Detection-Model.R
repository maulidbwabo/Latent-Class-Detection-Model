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
dynamicP=subset(dynamic,select = -c(1:14))
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
trainData = dynamicP[index, ]
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
             y = dynamicP$Region,
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
##Random Forest (Predictions on regressions)
#Training and Validation of the data
set.seed(1234)
index = createDataPartition(d$RegionArusha, p = 0.8, list = FALSE)
trainData = d[index, ]
validationData = d[-index, ]
##
#confirm structure
str(trainData[,c(4:8,9:14)])
#PCA
pc = prcomp(trainData[,c(4:8,9:14)], center = TRUE, scale. = TRUE)
summary(pc)
plot(pc)
##Predicting 
pcValidationData1 = predict(pc, newdata = validationData[,c(4:8,9:14)])
#scalable method using PcaMethods
pc=pca(trainData[,c(4:8,9:14)], method = "svd",nPcs = 4, scale = "uv", 
        center = TRUE)
pc
summary(pc)
##Validations two
pcValidationData2 = predict(pc, newdata = validationData[,c(4:8,9:14)])
pcValidationData1[,1]
pcValidationData2$score
pcValidationData2$scores[,1]
##The result turn out to be the same 
svmDataTrain = trainData[,.(KS3, E1)]
svmDataValidate = validationData[,.(KS3, E1)]
p1 = ggplot(data = svmDataTrain,
             aes(x =E1, y = KS3))
## data point colored by country
# set up training & validation data
svmDataTrain = trainData[,-1]
svmDataTrain
svmDataValidation = validationData[,-1]
##
set.seed(12345)
svmLinear =train(x = svmDataTrain,
                   y = trainData$RegionTanga,
                   method = "svmLinear",
                   preProcess = c("scale", "center", "pca"),
                   metric = "Accuracy",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            seeds = c(123, 234, 345, 456, 567, 678)
                   )
)
svmLinear
#run linear SVM on the full data set
p1 + geom_point(aes(colour = trainData$MC3)) +
  scale_colour_viridis(discrete = TRUE)
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
##Random Forest(Machine Learning)
library(randomForest)
library(tidyverse)
library(dslabs)
data("mnist_27")
str(mnist_27)
head(mnist_27)
##
library(randomForest)
dynamicP$Region=as.factor(dynamicP$Region)
str(dynamicP)
fit = randomForest(Region ~., data = dynamicP)
plot(fit)
##
dynamicP%>%
  mutate(y_hat = predict(fit, newdata = dynamicP)) %>%
  ggplot() +
  geom_point(aes(KS1, KS3)) +
  geom_line(aes(KS1, y_hat), col="red")
str(polls_2008)
mnist_27$test
#RANDOM FOREST
trainData
trainData$Region=as.factor(trainData$Region)
rfDataTrain = copy(trainData)
rfDataTrain
##Validation
validationData
rfDataValidation = copy(validationData)
##
set.seed(12345)
rfModel = train(x = rfDataTrain,
                 y = trainData$Region,
                 method = "ranger",
                 preProcess = c("scale", "center", "pca"),
                 metric = "Accuracy",
                 num.trees = 3,
                 trControl = trainControl(method = "cv",
                                          number = 3
                 )
)
rfMode
##Comparison between classification and Regression Tree (CART) and Random Forest 
library(data.table)
dynamic=read.csv("C:/Users/bwabo/OneDrive/Desktop/Ph.D. Thesis/Data set.csv",header = TRUE,sep = ",",stringsAsFactors = FALSE)
head(dynamic)
View(dynamic)
##Principal components analysis(Exploratory Data Analysis)
dynamic = as.data.table(dynamic)
dynamicP=subset(dynamic,select = -c(1:4,6:14))
str(dynamicP)
head(dynamicP)
dynamicP$Region=as.factor(dynamicP$Region)
head(dynamicP)
rf1 = randomForest(Region ~ ., dynamicP, ntree=50, norm.votes=FALSE)
rf1
summary(rf1)
dyna.rf = randomForest(Region ~ ., dynamicP, ntree=50, norm.votes=FALSE)
dyna.rf = grow(dyna.rf, 50)
print(dyna.rf)
plot(dyna.rf)
dyna.rf$predicted
dyna.rf$type
dyna.rf$classes
dyna.rf$importanceSD
set.seed(1)
data(dynamicP)
dyna.rf <- randomForest(Region ~ ., dynamicP, keep.forest=FALSE)
plot(dyna.rf)
dyna.rf
##MSDP
set.seed(1)
dyna1.rf = randomForest(Region ~ ., dynamicP, proximity=TRUE,
                        keep.forest=FALSE)
MDSplot(dyna1.rf, dynamicP$Region)
## Using different symbols for the classes:
MDSplot(dyna1.rf, dynamicP$Region, palette=rep(1, 3), pch=as.numeric(dynamicP$Region))
##Predict plot 
data(iris)
set.seed(111)
ind = sample(2, nrow(dynamicP), replace = TRUE, prob=c(0.8, 0.2))
dyna2.rf = randomForest(Region ~ ., data=dynamicP[ind == 1,])
dyna2.pred = predict(dyna2.rf, dynamicP[ind == 2,])
dyna2.pred
table(observed = dynamicP[ind==2, "Region"], predicted = dyna2.pred)
##Split the data into testing and trainning 
ind=sample(2,nrow(dynamicP),replace = TRUE,prob = C(0.7.03))
ind = sample(2, nrow(dynamicP), replace = TRUE, prob=c(0.8, 0.2))
trainData = dynamicP[ind==1,]
testData = dynamicP[ind==2,]
##Generate the Random Forest
dyna3_rf=randomForest(Region ~., data = trainData,ntree=100,proximity=T)
dyna4_rf=randomForest(Region ~.,data = testData,ntree=100,proximity=T)
plot(dyna4_rf)
plot(dyna3_rf)
##Random Forest for Testing Data
dynapred=predict(dyna3_rf,newdata=testData)
table(dynapred,testData$Region)
plot(dyna3_rf)
dyna3_rf
dyna4_rf
summary(dyna3_rf)
##Confussion Matrix
CM = table(dynapred, testData$Region)
accuracy = (sum(diag(CM)))/sum(CM)
accuracy
##Extended Random Forest 
cartDataTrain = copy(trainData)
cartDataTrain[,Region:=as.numeric(Region)]
cartDataValidation = copy(testData)
cartDataValidation[,Region:=as.numeric(Region)]
##CART Model
set.seed(12345)
cartModel  train(x = cartDataTrain,
                 y = trainData$Region,
                 method = "rpart",
                 preProcess = c("scale", "center", "pca"),
                 metric = "Accuracy",
                 tuneLength = 10,
                 trControl = trainControl(method = "cv",
                                          number = 5
                 )
)
cartModel
summary(cartModel)
#Testing 
predictOnTrainT = predict(cartModel, newdata = cartDataTrain)
mean( predictOnTrainT == trainData$Region)
##Validations 
predictOnTestT = predict(cartModel, newdata = cartDataValidation)
mean(predictOnTestT == testData$Region)
##Random Forest 
rfDataTrain = copy(trainData)
rfDataTrain[,Region:=as.numeric(Region)]
rfDataValidation = copy(testData)
rfDataValidation[,Region:=as.numeric(Region)]
##
set.seed(12345)
rfModel = train(x = rfDataTrain,
                y = trainData$Region,
                method = "ranger",
                preProcess = c("scale", "center", "pca"),
                metric = "Accuracy",
                num.trees = 20,
                trControl = trainControl(method = "cv",
                                         number = 5
                )
)
rfModel
#Stochastic Gradient Boosting (SGB) 
sgbDataTrain = copy(trainData)
sgbDataTrain
sgbDataTrain[,Region:=as.numeric(Region)]
sgbDataValidation = copy(testData)
sgbDataValidation[,Region:=as.numeric(Region)]
##Creating a dummy variable  
ddum = dummyVars("~.", data = sgbDataTrain)
sgbDataTrain = data.table(predict(ddum, newdata = sgbDataTrain))
sgbDataValidation = data.table(predict(ddum, newdata = sgbDataValidation))
#
sgbModel = train(KS1 ~.,
                  data = sgbDataTrain,
                  method = "gbm",
                  preProcess = c("scale", "center"),
                  metric = "RMSE",
                  trControl = trainControl(method = "cv",
                                           number = 5
                  ),
                  tuneGrid = expand.grid(interaction.depth = 1:3,
                                         shrinkage = 0.1,
                                         n.trees = c(50, 100, 150),
                                         n.minobsinnode = 10),
                  verbose = FALSE
)
sgbModel
summary(sgbModel)
#comparison between the predicted and actual data
mean(stats::residuals(sgbModel)^2)
mean((predict(sgbModel, sgbDataValidation) -
        sgbDataValidation$KS1)^2)
##
explainSGBt = explain(sgbModel, label = "sgbt",
                       data = sgbDataTrain,
                       y = sgbDataTrain$KS1)
explainSGBv = explain(sgbModel, label = "sgbv",
                       data = sgbDataValidation,
                       y = sgbDataValidation$KS1)
##
performanceSGBt = model_performance(explainSGBt)
performanceSGBv = model_performance(explainSGBv)
##
plot_grid(
  plot(performanceSGBt, performanceSGBv),
  plot(performanceSGBt, performanceSGBv, geom = "boxplot"),
  ncol = 2)
##
importanceSGBt = variable_importance(explainSGBt)
importanceSGBv = variable_importance(explainSGBv)
plot(importanceSGBt, importanceSGBv)
##Variable of response function
library(cowplot)
library(randomForest)
library(DALEX)
install.packages("survxai")
library(survxai)
library(mlr)
library(xgboost)
library(breakDown)
## Surv package require the object to be initially explained with Surv package
##LATENT CLASS DETECTION (REBUS-PL)
library(plspm)
dynamicP= as.data.table(dynamicP)
head(dynamicP)
summary(dynamicP)
view(dynamicP)
class(dynamicP)
dynamic1=as.list(dynamicP)
print(dynamicP)
print(dynamic1)
view(dynamic1)
# The path matrix
Knowledge = rep(0,5)
Sensing =rep(0,5)
Cognition=rep(0,5)
Agile=c(1,1,1,0,0)
Sustainable=c(1,1,1,1,0)
# Matrix created by row binding
Dynamic_Path=rbind(Knowledge,Sensing,Cognition,Agile,Sustainable)
print(Dynamic_Path)
# List of blocks(outer model)
Dynamic_blocks=list(1:5, 6:11, 17:21, 22:26,39:43)
class(Dynamic_blocks)
# Vector of modes (reflective)
Dynamic_modes =rep("A",5)
class(Dynamic_modes)
# plot the inner matrix
innerplot(Dynamic_Path)
class(Dynamic_pls)
# apply plspm
Dynamic_pls =plspm(dynamicP,Dynamic_Path , Dynamic_blocks, modes = Dynamic_modes)
print(Dynamic_pls)
Dynamic_pls$boot
Dynamic_pls$path_coefs
Dynamic_pls$unidim
plot(Dynamic_pls, what="loadings")
plot(Dynamic_pls)
plot(Dynamic_pls,arr.pos=0.3)
Dynamic_pls$outer_model
summary(Dynamic_pls)
#Bootstrap validation
Dynamic_val=plspm(dynamicP,Dynamic_Path,Dynamic_blocks,modes = Dynamic_modes,scaling = NULL,scheme = "centroid",
                  scaled = TRUE,boot.val = TRUE,br=5000)
Dynamic_val$boot
Dynamic_val$model

#breaking down the global model 
#cluster 
## Then compute cluster analysis on residuals of global model
# hierarchical cluster analysis on the LV scores
Dynamic_hclus =hclust(dist(Dynamic_pls$scores), method ="ward.D")
# Plot Dendrogram
plot(Dynamic_hclus, xlab ="", sub ="", cex = 0.8)
abline(h = 40, col ="#bc014655", lwd = 4)
# cut tree to obtain 3 clusters
clusters =cutree(Dynamic_hclus, k =3)
#
table(clusters)
#Scores and membership into the data frame
# latent variable scores in data frame
dy_scores=as.data.frame(Dynamic_pls$scores)
##Adding cluster to data frame
# add clusters to data frame
dy_scores$Cluster=as.factor(clusters)
#Picturing out the data
# add clusters to data frame
head(dy_scores,n=10)
#
library(plyr)
# calculate cluster centroids
centroids = ddply(dy_scores, .(Cluster), summarise,
                  AvKnowlegde = mean(Knowledge), AvSensing = mean(Sensing),
                  AvAgile = mean(Agile), AvSustainable= mean(Sustainable))

print(centroids)
#apply REBUS
Dynamic_reb=rebus.pls(Dynamic_pls)
#Apply REBUS
Dy_clus = res.clus(Dynamic_pls)
## To complete REBUS, run iterative algorithm
rebus_dy = it.reb(Dynamic_pls, Dy_clus, nk=3,
                   stop.crit=0.005, iter.max=100)
# bootstrapped path coefficients
Governance_3_pls$boot

# summarized results
summary(Governance_3_pls)

# running bootstrap validation
Governace_3 =plspm(Governace_3, Governance_3_path,Governance_3_blocks, modes = Governance_3_mods,
                   boot.val = TRUE, br = 10,00)
# bootstrap results
Governance_3l$boot
##Load data
mode(dynamicP)
class(dynamicP)
storage.mode(dynamicP)
lapply(dynamicP[,1:5],class)
dynamic2=data.matrix(dynamicP)
mode(dynamic2)
# Calculate plspm

##
#Example 
library(readxl)
Data_set <- read_excel("Data set.xlsx")
str(Data_set)
View(Data_set)
dynamicX=subset(Data_set,select = -c(1:15))
str(dynamicX)
mode(dynamicX)
str(dynamicP)
dynamicZ= as.data.table(dynamicX)
str(dynamicZ)
##
dy_inner=matrix(c(0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0), 4, 4,byrow=TRUE)

dimnames(dy_inner)=list(c("Knowledge","Sensing", "Cognition","Sustainable"),
                        c("Knowledge","Sensing", "Cognition", "Sustainable"))
dy_outer=list(c(1,2,3,4,5),c(6,7,8,9,10,11),c(17,18,19,20,21),c(39,40,41,42,43))
dy_mod=c("A","A","A","A")
head(dynamicX)
dy_global=plspm(dynamicZ,dy_inner,dy_outer,modes = dy_mod)
dy_global
summary(dy_global)

rebus_dy=rebus.pls(dy_global,stop.crit = 0.005,iter.max = 100)
rebus_dy
rebus_dy$loadings
##Local Models 
# local plspm models
locs = local.models(dy_global, rebus_dy)
summary(locs)
summary(locs$loc.model.1)
summary(locs$loc.model.2)
summary(locs$loc.model.3)
##Apply the Rebus Test
# apply rebus.test
# apply rebus.test
dy_test = rebus.test(dy_global, rebus_dy)

