### Web Scraping & Machine Learning ###
### Part II: Supervised Learning I ###
### 02: Basics and CART ###

# Setup

# install.packages("ggmap")
# install.packages("GGally")
# install.packages("corrplot")
# install.packages("rpart")
# install.packages("partykit")
# install.packages("pdp")
# install.packages("caret")
library(ggmap)
library(GGally)
library(corrplot)
library(rpart)
library(partykit)
library(pdp)
library(caret)

load("bremen.Rdata")

## Split data in training and test set

set.seed(151)

train <- sample(1:nrow(bremen), 0.8*nrow(bremen))
bremen_test <- bremen[-train,]
bremen_train <- bremen[train,]

bremen_train_c <- subset(bremen_train, complete.cases(bremen_train))

## Some data exploration

map <- qmap("Bremen, Germany", zoom = 12, maptype = "hybrid")
map + geom_point(data=bremen_train_c, aes(x=lon, y=lat, color=rent), size=2, alpha=0.5) + scale_color_gradientn(colours=rev(heat.colors(10))) 

map <- qmap("Bremerhafen, Germany", zoom = 12, maptype = "hybrid")
map + geom_point(data=bremen_train_c, aes(x=lon, y=lat, color=rent), size=2, alpha=0.5) + scale_color_gradientn(colours=rev(heat.colors(10))) 

ggpairs(bremen_train_c[,c(2:4,10)], lower = list(continuous = "smooth"))

cormtrx <- cor(bremen_train_c[,c(2:4,10)])
corrplot.mixed(cormtrx)

## CART

# Grow and prune tree (1-SE rule)

set.seed(733)
f_tree <- rpart(rent ~ m2 + rooms + lon + lat + city + distance, data = bremen_train_c, cp = 0.005)
f_tree
printcp(f_tree)
plotcp(f_tree)

minx <- which.min(f_tree$cptable[,"xerror"])
minxse <- f_tree$cptable[minx,"xerror"] + f_tree$cptable[minx,"xstd"]
minse <- which.min(abs(f_tree$cptable[,"xerror"] - minxse))
mincp <- f_tree$cptable[minse,"CP"]

p_tree <- prune(f_tree, cp = mincp)
p_tree

# Variable Importance and Plots

prty_tree <- as.party(p_tree)
plot(prty_tree, gp = gpar(fontsize = 6))

varImp(p_tree)

pdp1 <- partial(p_tree, pred.var = "m2")
plotPartial(pdp1, rug = T, train = bremen_train, alpha = 0.3)

pdp2 <- partial(p_tree, pred.var = c("lat", "lon"))
plotPartial(pdp2, levelplot = F, drape = T, colorkey = F, screen = list(z = 30, x = -60))

# Prediction

y_tree <- predict(p_tree, newdata = bremen_test)
postResample(pred = y_tree, obs = bremen_test$rent)
