###### Setting working directory ######

setwd("F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping")

###### Reading of the dataset ########

pa=read.csv("FF_Susceptibility_Mapping.csv") # reading of presence-absence dataset of wildfires

#pa$Fire  = as.factor(pa$Fire) # turning target variable into categorical variable

#pa = subset(pa, select = -c(1,2,3,11,12) ) # dropping multicollinear variables

head(pa)# to observe the dataset profile
summary(pa)# summary of the dataset
str(pa)


        ####### Preparation of train-Test split ######

library(caret)
set.seed(123)

# splitting the dataset into traning-testing subsets
trainIndex = createDataPartition(pa$Fire, p = .70, 
                                 list = FALSE, 
                                 times = 1)

training = pa[ trainIndex,] #70% data for model training
testing= pa[-trainIndex,] #30% for model testing



################################ RANDOM FOREST #############################
library(randomForest)

mod_rf <- randomForest(Fire~., data = training, ntree=100, cv.fold = 10)


##################### SAVING the model to disk ##########################
setwd("F:/Forest Fire/Project/FF Susceptibility Zone Mapping/Load Model")
saveRDS(mod_rf, "./RF_model.rds")

#################### LOADING THE MODEL #################################
model <- readRDS("./RF_model.rds")


####### Model details ########
print(mod_rf)# Accuracy summary


############ Importance of the various predictors ###########
varImp(mod_rf) 
barplot(var_rf,
        main = "Variable Importance",
        xlab = "Importance (in %)",
        ylab = "Predictor Variables",
        names.arg = c("Aspect", "Prep", "Temp", "Wind", "NDVI", 
                      "Slope", "Soil C", "TreeCover", "Road", "Water"),
        col = "darkred",
        horiz = T) #make height and width 1000 x 1000

###############################################################
################## RF model on test data #####################
pred=predict(mod_rf, newdata=testing) #predict on the test data

############ ACCURACY ASSESSMENT ############
library(caret)
library(ggplot2)

pred_50 = ifelse(pred>0.5,1,0)
cm <- table(testing$Fire, pred_50)
confusionMatrix(cm)

#CM Plot
fourfoldplot(cm, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")


####################### Other Metrics #####################3
library(MLmetrics)
F1_Score(pred_50, testing$Fire) 
Accuracy(pred_50, testing$Fire) 

MAE(pred_50, testing$Fire) 

RMSE(pred_50, testing$Fire) 

Precision(testing$Fire, pred_50, positive = NULL) 

Recall(testing$Fire, pred_50, positive = NULL) 

AUC(pred_50, testing$Fire) 

####### Test the importance of individual predictors #######
partialPlot(mod_rf, training, TreeCover)
partialPlot(mod_rf, training, ProximityToRoadways1)
partialPlot(mod_rf, training, avgWind)
partialPlot(mod_rf, training, water)


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
rf_pred = predict(stck, mod_rf, format = "GTiff") #use predict to implement the RF model stored

###### Exporting prediction data as Raster map #############
writeRaster(rf_pred, 
            filename = "F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping\\Susceptibility Maps\\RF\\RF_2.tiff",
            format = "GTiff", 
            overwrite=TRUE)


