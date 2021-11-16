##The Paper one 
library(data.table)
dynamic=read.csv("C:/Users/bwabo/OneDrive/Desktop/Ph.D. Thesis/Data set.csv",header = TRUE,sep = ",",stringsAsFactors = FALSE)
head(dynamic)
View(dynamic)
##Principal components analysis(Exploratory Data Analysis)
dynamicP=subset(dynamic,select = -c(1:14))
head(dynamicP)
#
#CPA analysis
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
## Scores and loadings plot
loadings(pc)
loadings(pcN)
slplot(pc)
slplot(pcN)
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
