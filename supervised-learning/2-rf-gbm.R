### Web Scraping & Machine Learning ###
### Part III: Supervised Learning II ###
### 03: Random Forests and Boosting ###

# Setup

# install.packages("caret")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("gbm")
# install.packages("pdp")
library(caret)
library(rpart)
library(randomForest)
library(gbm)
library(pdp)

load("bremen.Rdata")

## Split data in training and test set

set.seed(151)

train <- sample(1:nrow(bremen), 0.8*nrow(bremen))
bremen_test <- bremen[-train,]
bremen_train <- bremen[train,]

bremen_train_c <- subset(bremen_train, complete.cases(bremen_train))
bremen_test_c <- subset(bremen_test, complete.cases(bremen_test))

## Random Forest

ctrl  <- trainControl(method = "cv",
                      number = 10)

grid <- expand.grid(mtry = 1:6)

set.seed(231)
rf <- train(rent ~ m2 + rooms + lon + lat + city + distance,
            data = bremen_train_c,
            method = "rf",
            trControl = ctrl,
            tuneGrid = grid,
            importance = TRUE)

rf
plot(rf)
plot(rf$finalModel)
varImp(rf)

# Inspect Forest

getTree(rf$finalModel, k = 1, labelVar = T)[1:20,]

pdp3 <- partial(rf, pred.var = "m2", ice = T, trim.outliers = T)
pdp4 <- partial(rf, pred.var = "distance", ice = T, trim.outliers = T)
p1 <- plotPartial(pdp3, rug = T, train = bremen_train_c, alpha = 0.3)
p2 <- plotPartial(pdp4, rug = T, train = bremen_train_c, alpha = 0.3)
grid.arrange(p1, p2, ncol = 2)

pdp5 <- partial(rf, pred.var = c("lat", "lon"))
plotPartial(pdp5, levelplot = F, drape = T, colorkey = F, screen = list(z = 30, x = -60))

## Boosting

grid <- expand.grid(interaction.depth = 1:6, 
                    n.trees = c(500, 1000, 1500), 
                    shrinkage = c(0.01, 0.005),
                    n.minobsinnode = 10)
grid

set.seed(231)
bm <- train(rent ~ m2 + rooms + lon + lat + city + distance,
            data = bremen_train_c,
            method = "gbm",
            trControl = ctrl,
            tuneGrid = grid)

bm
plot(bm)
varImp(bm)

## CART

grid <- expand.grid(maxdepth = 1:30)
                    
set.seed(231)
cart <- train(rent ~ m2 + rooms + lon + lat + city + distance,
              data = bremen_train_c,
              method = "rpart2",
              trControl = ctrl,
              tuneGrid = grid)

cart
plot(cart)
varImp(cart)

## Linear regression

set.seed(231)
reg <- train(rent ~ m2 + rooms + lon + lat + city + distance,
            data = bremen_train_c,
            method = "glm",
            trControl = ctrl)

reg
summary(reg)
varImp(reg)

## Comparison (rf, gbm, rpart & glm)

resamps <- resamples(list(RandomForest = rf,
                          Boosting = bm,
                          CART = cart,
                          Regression = reg))

resamps
summary(resamps)
bwplot(resamps, metric = c("RMSE", "Rsquared"), scales = list(relation = "free"), xlim = list(c(0, 350), c(0, 1)))
splom(resamps, metric = "RMSE")
splom(resamps, metric = "Rsquared")

difValues <- diff(resamps)
summary(difValues)

## Prediction

y_rf <- predict(rf, newdata = bremen_test_c)
y_bm <- predict(bm, newdata = bremen_test_c)
y_cart <- predict(cart, newdata = bremen_test_c)
y_reg <- predict(reg, newdata = bremen_test_c)

postResample(pred = y_rf, obs = bremen_test_c$rent)
postResample(pred = y_bm, obs = bremen_test_c$rent)
postResample(pred = y_cart, obs = bremen_test_c$rent)
postResample(pred = y_reg, obs = bremen_test_c$rent)
