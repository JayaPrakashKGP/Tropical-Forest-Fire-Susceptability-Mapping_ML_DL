###### Setting working directory ######

setwd("F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping")

# Reading & Exploring the Data
data = read.csv("FF_Susceptibility_Mapping.csv", header=T)
#data$Fire = factor(data$Fire, ordered = T)

head(data)
summary(data)
str(data)
View(data)

# Train-test Split
set.seed(222)
inp <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training_data <- data[inp==1, ]
test_data <- data[inp==2, ]
table(training_data$Fire)
table(test_data$Fire)

################## Creating NN
library(neuralnet)
set.seed(123)
nn_1 <- neuralnet(Fire~.,
               data = training_data,
               hidden = 5,
               err.fct = "ce",
               linear.output = F,
               act.fct = "logistic",
               lifesign = 'full',
               rep = 3,
               algorithm = "sag",
               stepmax = 10000000)

nn_2 <- neuralnet(Fire~.,
               data = training_data,
               hidden = c(5,3),
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 3,
               algorithm = "rprop+",
               threshold = 0.1,
               learningrate.limit = NULL,
               learningrate.factor = list(minus = 0.5, plus = 1.2),
               stepmax = 10000000)

nn_3 <- neuralnet(Fire~., data=training_data, hidden = c(10,3),
                  threshold = 0.01, stepmax = 10000000,
                  rep = 3, startweights = NULL,
                  learningrate.limit = NULL,
                  learningrate.factor =
                    list(minus = 0.5, plus = 1.2),
                  learningrate=NULL, lifesign = "none",
                  lifesign.step = 1000, algorithm = "rprop+",
                  err.fct = "sse", act.fct = "logistic",
                  linear.output = F, exclude = NULL,
                  constant.weights = NULL, likelihood = FALSE)

nn <- neuralnet(Fire~., data=training_data, hidden = 1, threshold = 0.01,
                stepmax = 1e+05, rep = 1, startweights = NULL,
                learningrate.limit = NULL, learningrate.factor = list(minus = 0.5,plus = 1.2), 
                learningrate = NULL, lifesign = "none",
                lifesign.step = 1000, algorithm = "rprop+", err.fct = "sse",
                act.fct = "logistic", linear.output = TRUE, exclude = NULL,
                constant.weights = NULL, likelihood = FALSE)


########################  H2o ############################
library(mlr)
library(data.table)
train <- setDT(training_data)
test <- setDT(test_data)

#load the package
require(h2o)

#start h2o
localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")

#load data on H2o
trainh2o <- as.h2o(train)
testh2o <- as.h2o(test)

#set variables
y <- "Fire"
x <- setdiff(colnames(trainh2o),y)

#train the model - without hidden layer
deepmodel <- h2o.deeplearning(x = x
                              ,y = y
                              ,training_frame = trainh2o
                              ,standardize = T
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 100
                              ,seed = 1
                              ,nfolds = 5
                              ,variable_importances = T)

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 10)
h2o.performance(deepmodel,xval = T)

deepmodel <- h2o.deeplearning(x = x      
                              ,y = y      
                              ,training_frame = trainh2o      
                              ,validation_frame = testh2o         
                              ,standardize = T        
                              ,model_id = "deep_model"        
                              ,activation = "Rectifier"       
                              ,epochs = 100       
                              ,seed = 1       
                              ,hidden = 5         
                              ,variable_importances = T)

h2o.performance(deepmodel,valid = T)

                          ################## PLOTS ####################

cols <- heat.colors(1)
garson(nn_1) +
     scale_y_continuous('Importance', limits = c(0,0.30)) + 
     scale_fill_gradientn(colours = 'blue')  
     #scale_colour_gradientn(colours = 'blue')

olden(nn_1)
plotnet(nn_1)
neuralweights(nn_1)
lekprofile(nn_1)
layer_data(nn_1)



##################### SAVING the model to disk ##########################
setwd("F:/Forest Fire/Project/FF Susceptibility Zone Mapping/Load Model")
saveRDS(nn_1, "./NN_1.rds")

#################### LOADING THE MODEL #################################
model <- readRDS("./final_model.rds")

#plot our neural network 
plot(model, rep = 2)
plot(nn_1, rep=1)

# error
model$result.matrix
nn_1$result.matrix

########################### PREDICTION ##################################
library(neuralnet)
# confusion Matrix 
library(caret)
output <- predict(nn_1, rep=3, test_data[, -1], type="raw")
p1 <- output$net.result
pred_50 <- ifelse(p1 > 0.5, 1, 0)
cm <- table(test_data$Fire, pred_50)
confusionMatrix(cm)

################ sag ####################
output <- predict(nn_1, rep=3, test_data[, -1], type="raw")
pred <- ifelse(output > 0.5, 1, 0)
cm <- table(test_data$Fire, pred)
confusionMatrix(cm) #Acc


str(test_data$Fire)
str(output)

#CM Plot
fourfoldplot(cm, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

####################### Other Metrics #####################3
library(MLmetrics)
F1_Score(pred, test_data$Fire) 

Accuracy(pred, test_data$Fire) 

MAE(pred, test_data$Fire) #Mean Absolute Error Loss

Precision(test_data$Fire, pred, positive = NULL) 
Recall(test_data$Fire, pred, positive = NULL) 
RMSE(pred, test_data$Fire) 

AUC(pred, test_data$Fire) 
##################################### AUC_ROC ###################################
output <- predict(nn_1, rep=2, test_data[, -1], type="raw")

library(pROC)

roc = pROC::roc(test_data[,"Fire"], pred) #compare testing data
auc= pROC::auc(roc)
auc #Area under the curve:

##### Plotting the ROC curve #######
plot(roc)
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))


## This might not be necessary:
detach(package:nnet,unload = T)

library(ROCR)

## train.labels:= A vector, matrix, list, or data frame containing the true  
## class labels. Must have the same dimensions as 'predictions'.
str(output)
train_labels = test_data[,1]
str(train.labels)

## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
pred = prediction(output, train_labels)
perf = performance(pred, "tpr", "fpr")
plot(perf, lwd=2, col="blue", main="ROC - Title")
abline(a=0, b=1)


#####################################################################
############ Read as Raster predictors files ############# 
setwd("F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping\\Data")
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
dl_pred = predict(stck, nn_1, rep=2, format = "GTiff") #use predict to implement the model stored

###### Exporting prediction data as Raster map #############
writeRaster(dl_pred, 
            filename = "F:\\Forest Fire\\Project\\FF Susceptibility Zone Mapping\\Susceptibility Maps\\DL\\new2.tiff",
            format = "GTiff", 
            overwrite=TRUE)

getwd()

############## NN
nn_4 <- neuralnet(Fire~.,
                  data = training_data,
                  hidden = 10,
                  err.fct = "ce",
                  linear.output = F,
                  lifesign = 'full',
                  rep = 2,
                  algorithm = "rprop+",
                  stepmax = 10000000) #rep=2

nn_1 <- neuralnet(Fire~.,
                  data = training_data,
                  hidden = c(5,3),
                  err.fct = "ce",
                  act.fct = "logistic",
                  linear.output = F,
                  lifesign = 'full',
                  rep = 3,
                  algorithm = "rprop+",
                  stepmax = 10000000) #rep2
