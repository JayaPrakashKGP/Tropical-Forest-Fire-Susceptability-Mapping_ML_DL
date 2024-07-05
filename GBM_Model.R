
###### Setting working directory ######

setwd("F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping\\Data")

###### Reading & Exploring the dataset ########

data=read.csv("FF_Susceptibility_Mapping.csv") # reading of presence-absence dataset of wildfires

head(data)# to observe the dataset profile
summary(data)# summary of the dataset
str(data)

####### Preparation of train-Test split ######

library(caret)
set.seed(100)

# splitting the dataset into traning-testing subsets
trainIndex = createDataPartition(data$Fire, p = .70, 
                                 list = FALSE, 
                                 times = 1)

training = data[ trainIndex,] #70% data for model training
testing= data[-trainIndex,] #30% for model testing

###################### XGBoost Model ########################
library(gbm)

mod_gbm = gbm(Fire~.,
              data = training,
              distribution = "gaussian",
              cv.folds = 10,
              shrinkage = .1,
              n.minobsinnode = 10,
              n.trees = 200)

?gbm

mod_gbm = gbm(Fire~.,
              data = training,
              distribution = "bernoulli",
              cv.folds = 10,
              shrinkage = .1,
              n.minobsinnode = 10,
              n.trees = 200)

summary(mod_gbm, 
        cBars = TRUE,
        n.trees = mod_gbm$n.trees, 
        plotit = FALSE, 
        order = TRUE,
        method = relative.influence, 
        normalize = TRUE)


####### Model details ########

summary(mod_gbm)# variable importance

print(mod_gbm)# model summary

A gradient boosted model with gaussian loss function.
200 iterations were performed.
The best cross-validation iteration was 188.
There were 10 predictors of which 10 had non-zero influence'''


########### Test the model #####################

pred = predict.gbm(object = mod_gbm,
                   newdata = testing,
                   n.trees = 200,
                   type = "response")
library(caret)
library(ggplot2)

pred_60 = ifelse(pred>0.6,1,0)
cm <- table(testing$Fire, pred_60)
confusionMatrix(cm)

pred_50 = ifelse(pred>0.5,1,0)
cm <- table(testing$Fire, pred_50)
confusionMatrix(cm)


####################### Other Metrics #####################3
library(MLmetrics)
f1 = F1_Score(pred_60, testing$Fire)
print(f1)


Accuracy(pred_50, testing$Fire) #0.8712644

MAE(pred_50, testing$Fire) #Mean Absolute Error Loss

Precision(testing$Fire, pred_50, positive = NULL) 

Recall(testing$Fire, pred_50, positive = NULL) 
RMSE(pred_50, testing$Fire) 

AUC(pred_50, testing$Fire) 

#CM Plot
fourfoldplot(cm, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

################## ROC and AUC ###################
library(pROC)

roc = pROC::roc(testing[,"Fire"], pred) #compare testing data
auc= pROC::auc(roc)
auc #Area under the curve

##### Plotting the ROC curve #######
plot(roc)
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))


#####################################################################
############ Read as Raster predictors files ############# 
library(raster)

datafiles = Sys.glob("*.tif") # Construction of a dataframe for the loading of feature rasters in GeoTiff format
datafiles #list of predictors

stck = stack() #empty raster stack for storing raster layers

# loading of feature rasters to the dataframe
for(i in 1:NROW(datafiles)){
  tempraster = raster(datafiles[i])
  stck = stack(stck,tempraster)
}
names(stck)

###### preparation of RF model using training data ########
gbm_pred = predict(stck, mod_gbm, format = "GTiff") #use predict to implement the RF model stored

###### Exporting prediction data as Raster map #############
writeRaster(gbm_pred, 
            filename = "F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping\\Susceptibility Maps\\GBM\\GBM.tiff",
            format = "GTiff", 
            overwrite=TRUE)


