library(dplyr)
library(plyr)
library(ranger)
library(caret)
library(prospectr)
library(mpmi)
library(raster)
library(inflection)

setwd("/home/preston/OneDrive/Papers/Soil_remote_sensing")
setwd('C:\\Users\\prest\\OneDrive\\Papers\\Soil_remote_sensing')

eia=read.csv('EIA_soil_predictors_27Jan2021_PS.csv')
eia=eia[,-1]

eia$Soil.Subgr=as.factor(eia$Soil.Subgr)
eia$grt.grp=as.factor(eia$grt.grp)
eia$Order=as.factor(eia$Order)

#create train test splits
#using kennard stone algorithm
splits=kenStone(eia[,11:81], k=round(nrow(eia)*0.75), metric = 'mahal')

eia_train=eia[splits$model,]
eia_test=eia[splits$test,]

#create subsets for different levels of precision
eia_train_sg=eia_train[,c(5,11:81)]
eia_train_gg=eia_train[,c(6,7,11:81)]
eia_train_order=eia_train[,c(7,11:81)]
eia_train_class=eia_train[,c(4,11:81)]
eia_train_series=eia_train[,c(8,11:81)]

eia_test_sg=eia_test[,c(5,11:81)]
eia_test_gg=eia_test[,c(6,7,11:81)]
eia_test_order=eia_test[,c(7,11:81)]
eia_test_class=eia_test[,c(4,11:81)]
eia_test_series=eia_test[,c(8,11:81)]


################hierarchical####################

#balance case weights
#balance at the order level. Too many groups and way to unbalanced for finer levels of precision
w <- 1/table(eia_train_order$Order)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_order))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_order$Order==temp_label] <- w[i]
}

#######################order######################
table(eia_train$Order)
table(eia_test$Order)

#feature selection
order_cor=findCorrelation(cor(eia_train_order[,-1]), cutoff=0.9)
order_cor
order_cor=order_cor+1

eia_train_order_sub=eia_train_order[,-order_cor]

model=ranger(Order~., data=eia_train_order_sub, importance='impurity', case.weights = weights)
order_var=names(sort(importance(model), decreasing=TRUE))
eia_train_order_sub=eia_train_order[,c("Order", order_var)]

val=vector('list')
order_features=vector("list")
x=0
repeat{
  order_var=names(sort(importance(model), decreasing=TRUE))
  order_var=c('Order', order_var)
  temp=eia_train_order_sub[,1:(length(order_var)-1)]
  model=ranger(Order~., data=temp, importance='impurity', case.weights = weights)
  val=c(val, model$prediction.error)
  x=x+1
  order_features[[x]]=names(sort(importance(model), decreasing=TRUE))
  if(x==46){
    break
  }
}
val=unlist(val)
#val=data.frame(terrain_var[-1], val)
a=1:length(val)
plot(val~a)
val_order=data.frame(val, a)

#find curve inflection point
#order_infl=nls(val ~ SSasymp(a, yf, y0, log_alpha), data = val_order)
fit = lm(val ~ exp(a))
plot(a,val)
lines(a,fitted(fit),col="blue")


#find inflection
order_infl=round(fitted(fit),3)
diff(order_infl)
#no improvement after 38

variables_order=order_features[[38]]
variables_order=c("Order", variables_order)

#final model
eia_train_order=eia_train_order[,variables_order]
model_order=ranger(Order~., data=eia_train_order, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_order
sort(importance(model_order), decreasing=TRUE)

pred_order=predict(model_order, data=eia_test)
pred_order=pred_order$predictions

second_highest = colnames(pred_order)[apply(pred_order, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_order_prob=apply(pred_order, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_order_prob2=apply(pred_order, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_order=colnames(pred_order)[max.col(pred_order, ties.method="first")]
pred_order=as.factor(pred_order)
second_highest_order=as.factor(second_highest)

pred_order=factor(pred_order, levels=union(levels(eia_test$Order), levels(pred_order)))
second_highest_order=factor(second_highest_order, levels=union(levels(eia_test$Order), levels(second_highest_order)))

confusionMatrix(eia_test$Order, pred_order)
confusionMatrix(eia_test$Order, second_highest_order)

table(eia_test$Order)
table(pred_order)


#######################great group######################
table(eia_train$grt.grp)
table(eia_test$grt.grp)

#feature selection

#terrain
gg_cor=findCorrelation(cor(eia_train_gg[,-c(1:2)]), cutoff=0.9)
gg_cor
gg_cor=gg_cor+2

eia_train_gg_sub=eia_train_gg[,-gg_cor]
eia_train_gg_sub=eia_train_gg_sub[,-2]

model=ranger(grt.grp~., data=eia_train_gg_sub, importance='impurity', case.weights = weights)
gg_var=names(sort(importance(model), decreasing=TRUE))
eia_train_gg_sub=eia_train_gg[,c("grt.grp", gg_var)]

val=vector('list')
gg_features=vector("list")
x=0
repeat{
  gg_var=names(sort(importance(model), decreasing=TRUE))
  gg_var=c('grt.grp', gg_var)
  temp=eia_train_gg_sub[,1:(length(gg_var)-1)]
  model=ranger(grt.grp~., data=temp, importance='impurity', case.weights = weights)
  val=c(val, model$prediction.error)
  x=x+1
  gg_features[[x]]=names(sort(importance(model), decreasing=TRUE))
  if(x==46){
    break
  }
}
val=unlist(val)
#val=data.frame(terrain_var[-1], val)
a=1:length(val)
plot(val~a)
val_gg=data.frame(val, a)

#find curve inflection point
#gg_infl=nls(val ~ SSasymp(a, yf, y0, log_alpha), data = val_gg)
fit = lm(val ~ exp(a))
plot(a,val)
lines(a,fitted(fit),col="blue")


#find inflection
gg_infl=round(fitted(fit),3)
diff(gg_infl)


#no improvement after 39

variables_gg=gg_features[[39]]
variables_gg=c("grt.grp", variables_gg)

#no improvement after first 10 bands

variables_gg=c(variables_gg, 'Order')

#final model
eia_train_gg=eia_train_gg[,variables_gg]

#build order specific models

eia_train_gg_brun=eia_train_gg[eia_train_gg$Order=="BRUN",]
eia_train_gg_gley=eia_train_gg[eia_train_gg$Order=="GLEY",]
eia_train_gg_luv=eia_train_gg[eia_train_gg$Order=="LUV",]
eia_train_gg_org=eia_train_gg[eia_train_gg$Order=="ORG",]

eia_train_gg_brun=eia_train_gg_brun[,-c(10)]
eia_train_gg_gley=eia_train_gg_gley[,-c(10)]
eia_train_gg_luv=eia_train_gg_luv[,-c(10)]
eia_train_gg_org=eia_train_gg_org[,-c(10)]

eia_train_gg_brun$grt.grp=as.character(eia_train_gg_brun$grt.grp)
eia_train_gg_gley$grt.grp=as.character(eia_train_gg_gley$grt.grp)
eia_train_gg_luv$grt.grp=as.character(eia_train_gg_luv$grt.grp)
eia_train_gg_org$grt.grp=as.character(eia_train_gg_org$grt.grp)

#separate test data by predicted order not actual order, otherwise biasing results
eia_test_gg=data.frame(pred_order, eia_test_gg)

#model importance
model_gg=ranger(grt.grp~., data=eia_train_gg[,-10], importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
importance(model_gg)


eia_test_gg_brun=eia_test_gg[eia_test_gg$pred_order=="BRUN",]
eia_test_gg_gley=eia_test_gg[eia_test_gg$pred_order=="GLEY",]
eia_test_gg_luv=eia_test_gg[eia_test_gg$pred_order=="LUV",]
eia_test_gg_org=eia_test_gg[eia_test_gg$pred_order=="ORG",]

#set case weights
#try balancing by dominant vs not dominant because rare soils are getting overweighted in the predictions
#brunisol
w <- 1/table(eia_train_gg_brun$grt.grp)
w= w^(1/4)
w <- w/sum(w)

weights <- rep(0, nrow(eia_train_gg_brun))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_gg_brun$grt.grp==temp_label] <- w[i]
}

model_gg_brun=ranger(as.factor(grt.grp)~., data=eia_train_gg_brun, importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_gg_brun
sort(importance(model_gg_brun), decreasing=TRUE)

pred_gg_brun=predict(model_gg_brun, data=eia_test_gg_brun,probability=TRUE, splitrule = "extratrees")
pred_gg_brun=pred_gg_brun$predictions

second_highest = colnames(pred_gg_brun)[apply(pred_gg_brun, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_gg_brun_prob=apply(pred_gg_brun, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_gg_brun_prob2=apply(pred_gg_brun, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_brun=colnames(pred_gg_brun)[max.col(pred_gg_brun, ties.method="first")]
pred_gg_brun=as.factor(pred_gg_brun)
second_highest_brun_gg=as.factor(second_highest)

#gleysol
w <- 1/table(eia_train_gg_gley$grt.grp)
w= w^(1/4)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_gg_gley))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_gg_gley$grt.grp==temp_label] <- w[i]
}

model_gg_gley=ranger(as.factor(grt.grp)~., data=eia_train_gg_gley, importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_gg_gley
sort(importance(model_gg_gley), decreasing=TRUE)

pred_gg_gley=predict(model_gg_gley, data=eia_test_gg_gley,probability=TRUE, splitrule = "extratrees")
pred_gg_gley=pred_gg_gley$predictions

second_highest = colnames(pred_gg_gley)[apply(pred_gg_gley, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_gg_gley_prob=apply(pred_gg_gley, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_gg_gley_prob2=apply(pred_gg_gley, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_gley=colnames(pred_gg_gley)[max.col(pred_gg_gley, ties.method="first")]
pred_gg_gley=as.factor(pred_gg_gley)
second_highest_gley_gg=as.factor(second_highest)

#luvisol
w <- 1/table(eia_train_gg_luv$grt.grp)
w= w^(1/4)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_gg_luv))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_gg_luv$grt.grp==temp_label] <- w[i]
}

model_gg_luv=ranger(as.factor(grt.grp)~., data=eia_train_gg_luv, importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_gg_luv
sort(importance(model_gg_luv), decreasing=TRUE)

pred_gg_luv=predict(model_gg_luv, data=eia_test_gg_luv,probability=TRUE, splitrule = "extratrees")
pred_gg_luv=pred_gg_luv$predictions

second_highest = colnames(pred_gg_luv)[apply(pred_gg_luv, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_gg_luv_prob=apply(pred_gg_luv, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_gg_luv_prob2=apply(pred_gg_luv, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_luv=colnames(pred_gg_luv)[max.col(pred_gg_luv, ties.method="first")]
pred_gg_luv=as.factor(pred_gg_luv)
second_highest_luv_gg=as.factor(second_highest)

#organic
w <- 1/table(eia_train_gg_org$grt.grp)
w= w^(1/4)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_gg_org))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_gg_org$grt.grp==temp_label] <- w[i]
}

model_gg_org=ranger(as.factor(grt.grp)~., data=eia_train_gg_org, importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_gg_org
sort(importance(model_gg_org), decreasing=TRUE)

pred_gg_org=predict(model_gg_org, data=eia_test_gg_org,probability=TRUE, splitrule = "extratrees")
pred_gg_org=pred_gg_org$predictions

second_highest = colnames(pred_gg_org)[apply(pred_gg_org, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_gg_org_prob=apply(pred_gg_org, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_gg_org_prob2=apply(pred_gg_org, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_org=colnames(pred_gg_org)[max.col(pred_gg_org, ties.method="first")]
pred_gg_org=as.factor(pred_gg_org)
second_highest_org_gg=as.factor(second_highest)

#combine results
pred_gg=as.factor(c(as.character(pred_gg_brun), as.character(pred_gg_gley), as.character(pred_gg_luv), as.character(pred_gg_org)))
second_highest_gg=as.factor(c(as.character(second_highest_brun_gg), as.character(second_highest_gley_gg), as.character(second_highest_luv_gg), as.character(second_highest_org_gg)))
actual_gg=as.factor(c(as.character(eia_test_gg_brun$grt.grp),as.character(eia_test_gg_gley$grt.grp),as.character(eia_test_gg_luv$grt.grp), as.character(eia_test_gg_org$grt.grp)))

pred_gg=factor(pred_gg, levels=union(levels(actual_gg), levels(pred_gg)))
second_highest_gg=factor(second_highest_gg, levels=union(levels(actual_gg), levels(second_highest_gg)))

confusionMatrix(actual_gg, pred_gg)
confusionMatrix(actual_gg, second_highest_gg)

table(eia_test$grt.grp)
table(pred_gg)



######################subgroup##########################
#balance case weights
#balance at the order level. Too many groups and way to unbalanced for finer levels of precision
w <- 1/table(eia_train_order$Order)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_order))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_order$Order==temp_label] <- w[i]
}


table(eia_train$Soil.Subgr)
table(eia_test$Soil.Subgr)
#feature selection

sg_cor=findCorrelation(cor(eia_train_sg[,-1]), cutoff=0.9)
sg_cor
sg_cor=sg_cor+1

eia_train_sg_sub=eia_train_sg[,-sg_cor]

model=ranger(Soil.Subgr~., data=eia_train_sg_sub, importance='impurity', case.weights = weights)
sg_var=names(sort(importance(model), decreasing=TRUE))
eia_train_sg_sub=eia_train_sg[,c("Soil.Subgr", sg_var)]

val=vector('list')
sg_features=vector("list")
x=0
repeat{
  sg_var=names(sort(importance(model), decreasing=TRUE))
  sg_var=c('sg', sg_var)
  temp=eia_train_sg_sub[,1:(length(sg_var)-1)]
  model=ranger(Soil.Subgr~., data=temp, importance='impurity', case.weights = weights)
  val=c(val, model$prediction.error)
  x=x+1
  sg_features[[x]]=names(sort(importance(model), decreasing=TRUE))
  if(x==46){
    break
  }
}
val=unlist(val)
#val=data.frame(terrain_var[-1], val)
a=1:length(val)
plot(val~a)
val_sg=data.frame(val, a)

#find curve inflection point
#sg_infl=nls(val ~ SSasymp(a, yf, y0, log_alpha), data = val_sg)
fit = lm(val ~ exp(a))
plot(a,val)
lines(a,fitted(fit),col="blue")


#find inflection
sg_infl=round(fitted(fit),3)
diff(sg_infl)
#no improvement after 39

variables_sg=sg_features[[39]]
variables_sg=c("Soil.Subgr", variables_sg)


#final model
eia_train_sg=eia_train_sg[,variables_sg]

#build models for each great group
#loop through to build models
eia_train$grt.grp=as.character(eia_train$grt.grp)
eia_train$Soil.Subgr=as.character(eia_train$Soil.Subgr)


#order pred_gg to match eia_test data order
nh=c(eia_test_gg_brun$Norm_Height,eia_test_gg_gley$Norm_Height, eia_test_gg_luv$Norm_Height, eia_test_gg_org$Norm_Height)

pred_gg_=data.frame(pred_gg, nh)
colnames(pred_gg_)=c("pred_gg", 'Norm_Height')

eia_test=eia_test[,-82]

eia_test=merge(eia_test, pred_gg_, by='Norm_Height')

eia_test=eia_test[!duplicated(eia_test),]
eia_test=eia_test[!duplicated(eia_test$field_1),]

#feature importance
#model importance
eia_train_sg=eia_train[,variables_sg]
eia_train_sg$Soil.Subgr=as.factor(eia_train_sg$Soil.Subgr)
model_sg=ranger(Soil.Subgr~., data=eia_train_sg, importance='impurity', probability=TRUE, splitrule = "extratrees", case.weights = weights)
importance(model_sg)


#predict results
predictions=data.frame(pred_sg=as.character(), second_highest_sg=as.character(), actual=as.character())
models_subgroup=vector('list')
y=0
for (x in unique(eia_train$grt.grp)){
  try({
  y=y+1
  eia_train_sub=eia_train[eia_train$grt.grp==x,]
  eia_train_sub=eia_train_sub[,variables_sg]
  w <- 1/table(eia_train_sub$Soil.Subgr)
  w <- w^(1/4)
  w <- w/sum(w)
  weights <- rep(0, nrow(eia_train_sub))
  for (i in 1:length(w)){
    temp_label=names(w)[i]
    weights[eia_train_sub$Soil.Subgr==temp_label] <- w[i]
  }
  eia_train_sub$Soil.Subgr=as.factor(eia_train_sub$Soil.Subgr)
  model_sg=ranger(Soil.Subgr~., data=eia_train_sub, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)
  models_subgroup[[y]]=model_sg
  eia_test_sub=eia_test[eia_test$pred_gg==x,]
  
  pred_sg=predict(model_sg, data=eia_test_sub)
  pred_sg=pred_sg$predictions
  
  second_highest = colnames(pred_sg)[apply(pred_sg, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
  pred_sg_prob=apply(pred_sg, 1, function(x) sort(x, decreasing=TRUE)[1])
  pred_sg_prob2=apply(pred_sg, 1, function(x) sort(x, decreasing=TRUE)[2])
  pred_sg=colnames(pred_sg)[max.col(pred_sg, ties.method="first")]
  second_highest_sg=as.character(second_highest)
  actual=as.character(eia_test_sub$Soil.Subgr)
  sg_temp=data.frame(pred_sg, second_highest_sg, actual)
  predictions=rbind(predictions, sg_temp)
  })
}

pred_sg=as.factor(predictions$pred_sg)
second_highest_sg=as.factor(predictions$second_highest_sg)
actual=as.factor(predictions$actual)

pred_sg=factor(pred_sg, levels=union(levels(actual), levels(pred_sg)))
second_highest_sg=factor(second_highest_sg, levels=union(levels(actual), levels(second_highest_sg)))

confusionMatrix(actual, pred_sg)
confusionMatrix(actual, second_highest_sg)

table(eia_test$Soil.Subgr)
table(pred_sg)

####################unconstrained##################
w <- 1/table(eia_train_order$Order)
w <- w/sum(w)
weights <- rep(0, nrow(eia_train_order))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_train_order$Order==temp_label] <- w[i]
}


eia_train_gg
eia_train_sg

model_gg_unconstrained=ranger(grt.grp~., data=eia_train_gg[,-10], importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)
model_sg_unconstrained=ranger(Soil.Subgr~., data=eia_train_sg, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)

#gg
pred_gg_unconstrained=predict(model_gg_unconstrained, data=eia_test)
pred_gg_unconstrained=pred_gg_unconstrained$predictions

second_highest = colnames(pred_gg_unconstrained)[apply(pred_gg_unconstrained, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_gg_unconstrained_prob=apply(pred_gg_unconstrained, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_gg_unconstrained_prob2=apply(pred_gg_unconstrained, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_unconstrained=colnames(pred_gg_unconstrained)[max.col(pred_gg_unconstrained, ties.method="first")]
pred_gg_unconstrained=as.factor(pred_gg_unconstrained)
second_highest_gg_unconstrained=as.factor(second_highest)

pred_gg_unconstrained=factor(pred_gg_unconstrained, levels=union(levels(eia_test$grt.grp), levels(pred_gg_unconstrained)))
second_highest_gg_unconstrained=factor(second_highest_gg_unconstrained, levels=union(levels(eia_test$grt.grp), levels(second_highest_gg_unconstrained)))

confusionMatrix(eia_test$grt.grp, pred_gg_unconstrained)
confusionMatrix(eia_test$grt.grp, second_highest_gg_unconstrained)

#sg
pred_sg_unconstrained=predict(model_sg_unconstrained, data=eia_test)
pred_sg_unconstrained=pred_sg_unconstrained$predictions

second_highest = colnames(pred_sg_unconstrained)[apply(pred_sg_unconstrained, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_sg_unconstrained_prob=apply(pred_sg_unconstrained, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_sg_unconstrained_prob2=apply(pred_sg_unconstrained, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_sg_unconstrained=colnames(pred_sg_unconstrained)[max.col(pred_sg_unconstrained, ties.method="first")]
pred_sg_unconstrained=as.factor(pred_sg_unconstrained)
second_highest_sg_unconstrained=as.factor(second_highest)

pred_sg_unconstrained=factor(pred_sg_unconstrained, levels=union(levels(eia_test$Soil.Subgr), levels(pred_sg_unconstrained)))
second_highest_sg_unconstrained=factor(second_highest_sg_unconstrained, levels=union(levels(eia_test$Soil.Subgr), levels(second_highest_sg_unconstrained)))

confusionMatrix(eia_test$Soil.Subgr, pred_sg_unconstrained)
confusionMatrix(eia_test$Soil.Subgr, second_highest_sg_unconstrained)

###############aggregate from bottom up#########################
#create conversion
gg_table=eia_train[,c(5,6)]
gg_table=unique(gg_table)

order_table=eia_train[,c(5,7)]
order_table=unique(order_table)
order_table=order_table[-42,]

##gg##
pred_gg_bottom=data.frame(pred_sg_unconstrained, 1:length(pred_sg_unconstrained))
colnames(pred_gg_bottom)=c('Soil.Subgr', 'sample')
pred_gg_bottom=merge(pred_gg_bottom, gg_table, by='Soil.Subgr')
pred_gg_bottom=pred_gg_bottom[order(pred_gg_bottom$sample),]
pred_gg_bottom=pred_gg_bottom$grt.grp

second_highest_gg_bottom=data.frame(second_highest_sg_unconstrained,1:length(second_highest_sg_unconstrained))
colnames(second_highest_gg_bottom)=c('Soil.Subgr', 'sample')
second_highest_gg_bottom=merge(second_highest_gg_bottom, gg_table, by='Soil.Subgr')
second_highest_gg_bottom=second_highest_gg_bottom[order(second_highest_gg_bottom$sample),]
second_highest_gg_bottom=second_highest_gg_bottom$grt.grp

pred_gg_bottom=factor(pred_gg_bottom, levels=union(levels(eia_test$grt.grp), levels(pred_gg_bottom)))
second_highest_gg_bottom=factor(second_highest_gg_bottom, levels=union(levels(eia_test$grt.grp), levels(second_highest_gg_bottom)))

confusionMatrix(eia_test$grt.grp, pred_gg_bottom)
confusionMatrix(eia_test$grt.grp, second_highest_gg_bottom)

##order##
pred_order_bottom=data.frame(pred_sg_unconstrained, 1:length(pred_sg_unconstrained))
colnames(pred_order_bottom)=c('Soil.Subgr', 'sample')
pred_order_bottom=merge(pred_order_bottom, order_table, by='Soil.Subgr')
pred_order_bottom=pred_order_bottom[order(pred_order_bottom$sample),]
pred_order_bottom=pred_order_bottom$Order

second_highest_order_bottom=data.frame(second_highest_sg_unconstrained,1:length(second_highest_sg_unconstrained))
colnames(second_highest_order_bottom)=c('Soil.Subgr', 'sample')
second_highest_order_bottom=merge(second_highest_order_bottom, order_table, by='Soil.Subgr')
second_highest_order_bottom=second_highest_order_bottom[order(second_highest_order_bottom$sample),]
second_highest_order_bottom=second_highest_order_bottom$Order

pred_order_bottom=factor(pred_order_bottom, levels=union(levels(eia_test$Order), levels(pred_order_bottom)))
second_highest_order_bottom=factor(second_highest_order_bottom, levels=union(levels(eia_test$Order), levels(second_highest_order_bottom)))

confusionMatrix(eia_test$Order, pred_order_bottom)
confusionMatrix(eia_test$Order, second_highest_order_bottom)




#########################create maps#####################
variables_sg
variables_gg
variables_order

roi=shapefile('/home/preston/OneDrive/Papers/Soil_remote_sensing/figures_roi.shp')
roi=roi[roi$id==1,]

sda=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_SDA.vrt')
Stand_Height=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_Stand_Height.vrt')
Slope_Height=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_Slope_Height.vrt')
ndvi_med_oct=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_oct/neab_sen2_ndvi_med_oct.vrt')
MRRTF=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_MRRTF.vrt')
MRVBF=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_MRVBF.vrt')
Valley_Depth=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_Valley_Depth.vrt')
ari=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ari_med/neab_sen2_ari_med.vrt')
ndvi_med=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med/neab_sen2_ndvi_med.vrt')
ndvi_max_sept=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_max_sept/neab_sen2_ndvi_max_sept.vrt')
ndvi_med_sept=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_sept/neab_sen2_ndvi_med_sept.vrt')
reip=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_reip_med/neab_sen2_reip_med.vrt')
ndvi_max_jul=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_max_jul/neab_sen2_ndvi_max_jul.vrt')
ndvi_med_apr=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_april/neab_sen2_ndvi_med_april.vrt')
rg_med=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_rg_med/neab_sen2_rg_med.vrt')

ndvi_med_oct=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_oct/neab_sen2_ndvi_med_oct.vrt')
swi=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_SWI.vrt')
ndvi_max_may=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_max_may/neab_sen2_ndvi_max_may.vrt')
b2_fall=stack('/media/preston/My Book/North_East_Alberta/neab_sen2_mosaic_fall/neab_sen2_mosaic_fall.vrt')
b2_fall=b2_fall[[2]]
vh_oct=stack('/media/preston/My Book/North_East_Alberta/neab_sen1_mosaic_oct/neab_sen1_mosaic_oct.vrt')
vv_oct=vh_oct[[1]]
vh_oct=vh_oct[[2]]
ndvi_med_may=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_may/neab_sen2_ndvi_med_may.vrt')
ndvi_max_may=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_max_may/neab_sen2_ndvi_max_may.vrt')
ndvi_med_jul=raster('/media/preston/My Book/North_East_Alberta/neab_sen2_ndvi_med_jul/neab_sen2_ndvi_med_jul.vrt')
vh=stack('/media/preston/My Book/North_East_Alberta/neab_sen1_mosaic/neab_sen1_mosaic.vrt')
vh=vh[[2]]
tri=raster('/media/preston/My Book/North_East_Alberta/LiDAR_Data/neab_tri.vrt')

#clip
Stand_Height=crop(Stand_Height, roi)
Slope_Height=crop(Slope_Height, roi)
MRRTF=crop(MRRTF, roi)
MRVBF=crop(MRVBF, roi)
Valley_Depth=crop(Valley_Depth, roi)
swi=crop(swi, roi)
tri=crop(tri, roi)
sda=crop(sda, roi)

#make sure resolution is the same
ndvi_med_oct=resample(crop(ndvi_med_oct, Slope_Height), Slope_Height)
ari=resample(crop(ari, Slope_Height), Slope_Height)
ndvi_med=resample(crop(ndvi_med, Slope_Height), Slope_Height)
ndvi_max_sept=resample(crop(ndvi_max_sept, Slope_Height), Slope_Height)
ndvi_med_sept=resample(crop(ndvi_med_sept, Slope_Height), Slope_Height)
b2_fall=resample(crop(b2_fall, Slope_Height), Slope_Height)
reip=resample(crop(reip, Slope_Height), Slope_Height)
ndvi_max_jul=resample(crop(ndvi_max_jul, Slope_Height), Slope_Height)
ndvi_med_apr=resample(crop(ndvi_med_apr, Slope_Height), Slope_Height)
rg_med=resample(crop(rg_med, Slope_Height), Slope_Height)

ndvi_med_oct=resample(crop(ndvi_med_oct, Slope_Height), Slope_Height)
ndvi_max_may=resample(crop(ndvi_max_may, Slope_Height), Slope_Height)
vh_oct=resample(crop(vh_oct, Slope_Height), Slope_Height)
vv_oct=resample(crop(vv_oct, Slope_Height), Slope_Height)
ndvi_med_may=resample(crop(ndvi_med_may, Slope_Height), Slope_Height)
ndvi_max_may=resample(crop(ndvi_max_may, Slope_Height), Slope_Height)
ndvi_med_jul=resample(crop(ndvi_med_jul, Slope_Height), Slope_Height)
vh=resample(crop(vh, Slope_Height), Slope_Height)


#create mask
plot(ndvi_med)
mask_rast=ndvi_med
mask_rast@data@values=ifelse(mask_rast@data@values<0.3, 1, 2)
plot(mask_rast)

#build models with only data that intersects with the site
#eia_sub=eia
#coordinates(eia_sub)=~Easting+Northing
#eia_sub=crop(eia_sub, roi)
#eia_sub=data.frame(eia_sub)

eia_sg=eia[,c('grt.grp',variables_sg)]
eia_gg=eia[,c(variables_gg)]
eia_order=eia[,variables_order]

#balance case weights
#balance at the order level. Too many groups and way to unbalanced for finer levels of precision
#predict at the order level
w <- 1/table(eia_order$Order)
w <- w/sum(w)
weights <- rep(0, nrow(eia_order))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_order$Order==temp_label] <- w[i]
}

model_order_figure=ranger(Order~., data=eia_order, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)

#great group models
eia_gg$grt.grp=as.character(eia_gg$grt.grp)
eia_gg$Order=as.character(eia_gg$Order)
models_gg=vector('list')
y=0
for (x in unique(eia_train$Order)){
  try({
    y=y+1
    eia_gg_sub=eia_gg[eia_gg$Order==x,]
    w <- 1/table(eia_gg_sub$grt.grp)
    w <- w^(1/4)
    w <- w/sum(w)
    weights <- rep(0, nrow(eia_gg_sub))
    for (i in 1:length(w)){
      temp_label=names(w)[i]
      weights[eia_gg_sub$grt.grp==temp_label] <- w[i]
    }
    eia_gg_sub=eia_gg_sub[,colnames(eia_gg_sub)!='Order',]
    model_gg=ranger(as.factor(grt.grp)~., data=eia_gg_sub, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)
    models_gg[[y]]=model_gg
  })
}

names(models_gg)=unique(eia_train$Order)
models_gg

#subgroup models
eia_sg$Soil.Subgr=as.character(eia_sg$Soil.Subgr)
eia_sg$grt.grp=as.character(eia_sg$grt.grp)
models_sg=vector('list')
y=0
for (x in unique(eia_train$grt.grp)){
  try({
    y=y+1
    eia_sg_sub=eia_sg[eia_sg$grt.grp==x,]
    w <- 1/table(eia_sg_sub$Soil.Subgr)
    w <- w^(1/4)
    w <- w/sum(w)
    weights <- rep(0, nrow(eia_sg_sub))
    for (i in 1:length(w)){
      temp_label=names(w)[i]
      weights[eia_sg_sub$Soil.Subgr==temp_label] <- w[i]
    }
    eia_sg_sub=eia_sg_sub[,-1]
    model_sg=ranger(as.factor(Soil.Subgr)~., data=eia_sg_sub, importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)
    models_sg[[y]]=model_sg
  })
}

names(models_sg)=unique(eia_train$grt.grp)

models_sg

#predict order for each point


#order
stack_order=stack(Stand_Height, MRVBF, Valley_Depth, vh_oct, ari, Slope_Height, ndvi_med_oct, MRRTF, b2_fall)
stack_order=rasterToPoints(stack_order)

coords=stack_order[,1:2]

stack_order=stack_order[,-c(1:2)]

colnames(stack_order)=variables_order[-1]

pred_order=predict(model_order_figure, stack_order)
pred_order=pred_order$predictions
second_highest_order = colnames(pred_order)[apply(pred_order, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_prob_order=apply(pred_order, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_prob_order2=apply(pred_order, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_order=colnames(pred_order)[max.col(pred_order, ties.method="first")]

pred_order1=data.frame(as.factor(pred_order), coords)
pred_order2=data.frame(as.factor(second_highest_order), coords)
pred_prob_order_1=data.frame(pred_prob_order, coords)
pred_prob_order_2=data.frame(pred_prob_order2, coords)

coordinates(pred_order1)=~x+y
coordinates(pred_order2)=~x+y
coordinates(pred_prob_order_1)=~x+y
coordinates(pred_prob_order_2)=~x+y

pred_order1=rasterFromXYZ(pred_order1)
pred_order2=rasterFromXYZ(pred_order2)
pred_prob_order_1=rasterFromXYZ(pred_prob_order_1)
pred_prob_order_2=rasterFromXYZ(pred_prob_order_2)

pred_order_rast=stack(pred_order1, pred_order2, pred_prob_order_1, pred_prob_order_2)
pred_order_rast=mask(pred_order_rast, mask_rast, maskvalue=1)
writeRaster(pred_order_rast, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_order.tif', format='GTiff')

write.csv(round(table(pred_order)/sum(table(pred_order))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/order_1_table_2022_04_06.csv')
write.csv(round(table(second_highest_order)/sum(table(second_highest_order))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/order_2_table_2022_04_06.csv')

plot(pred_order1)
round(table(pred_order)/sum(table(pred_order))*100,2)[1]
round(table(second_highest_order)/sum(table(second_highest_order))*100,2)[4]

#modal filter
pred_order1_filt=focal(pred_order1, w=matrix(1,3,3), fun=modal)
pred_order2_filt=focal(pred_order2, w=matrix(1,3,3), fun=modal)
pred_prob_order_1_filt=focal(pred_prob_order_1, w=matrix(1,3,3), fun=modal)
pred_prob_order_2_filt=focal(pred_prob_order_2, w=matrix(1,3,3), fun=modal)

pred_order_rast_filt=stack(pred_order1_filt, pred_order2_filt, pred_prob_order_1_filt, pred_prob_order_2_filt)
pred_order_rast_filt=mask(pred_order_rast_filt, mask_rast, maskvalue=1)
writeRaster(pred_order_rast_filt, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_order_filt.tif', format='GTiff')


#gg
variables_gg
stack_gg=stack(Stand_Height, MRVBF, Valley_Depth, vh_oct, ari, Slope_Height, MRRTF, ndvi_med_oct)
stack_gg=rasterToPoints(stack_gg)

coords=stack_gg[,1:2]

stack_gg=stack_gg[,-c(1:2)]

colnames(stack_gg)=variables_gg[-c(1, 11)]

pred_gg1=data.frame(pred_gg1=as.character(), x=as.numeric(), y=as.numeric())
pred_gg2=data.frame(pred_gg1=as.character(), x=as.numeric(), y=as.numeric())
pred_prob_gg_1=data.frame(pred_gg1=as.character(), x=as.numeric(), y=as.numeric())
pred_prob_gg_2=data.frame(pred_gg1=as.character(), x=as.numeric(), y=as.numeric())
for (x in unique(pred_order)){
  stack_gg_sub=stack_gg[pred_order==x,]
  coords_sub=coords[pred_order==x,]
  pred_gg_sub=predict(models_gg[[x]], stack_gg_sub)
  pred_gg=pred_gg_sub$predictions
  second_highest_gg = colnames(pred_gg)[apply(pred_gg, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
  pred_prob_gg=apply(pred_gg, 1, function(x) sort(x, decreasing=TRUE)[1])
  pred_prob_gg2=apply(pred_gg, 1, function(x) sort(x, decreasing=TRUE)[2])
  pred_gg=colnames(pred_gg)[max.col(pred_gg, ties.method="first")]
  pred_gg=data.frame(pred_gg, coords_sub)
  second_highest_gg=data.frame(second_highest_gg, coords_sub)
  pred_prob_gg1=data.frame(pred_prob_gg, coords_sub)
  pred_prob_gg2=data.frame(pred_prob_gg2, coords_sub)
  pred_gg1=rbind(pred_gg1, pred_gg)
  pred_gg2=rbind(pred_gg2, second_highest_gg)
  pred_prob_gg_1=rbind(pred_prob_gg_1, pred_prob_gg1)
  pred_prob_gg_2=rbind(pred_prob_gg_2, pred_prob_gg2)
}

pred_gg=pred_gg1$pred_gg
second_highest_gg=pred_gg2$second_highest_gg

pred_gg=pred_gg1$pred_gg
second_highest_gg=pred_gg2$second_highest_gg

coordinates(pred_gg1)=~x+y
coordinates(pred_gg2)=~x+y
coordinates(pred_prob_gg_1)=~x+y
coordinates(pred_prob_gg_2)=~x+y

pred_gg1=rasterFromXYZ(pred_gg1)
pred_gg2=rasterFromXYZ(pred_gg2)
pred_prob_gg_1=rasterFromXYZ(pred_prob_gg_1)
pred_prob_gg_2=rasterFromXYZ(pred_prob_gg_2)

pred_gg_rast=stack(pred_gg1, pred_gg2, pred_prob_gg_1, pred_prob_gg_2)
pred_gg_rast=mask(pred_gg_rast, mask_rast, maskvalue=1)
writeRaster(pred_gg_rast, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_gg.tif', format='GTiff')

write.csv(round(table(pred_gg)/sum(table(pred_gg))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/gg_1_table_2022_04_06.csv')
write.csv(round(table(second_highest_gg)/sum(table(second_highest_gg))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/gg_2_table_2022_04_06.csv')

plot(pred_gg1)
round(table(pred_gg)/sum(table(pred_gg))*100,2)[1]
round(table(second_highest_gg)/sum(table(second_highest_gg))*100,2)[3]

#median filter
pred_gg1_filt=focal(pred_gg1, w=matrix(1,3,3), fun=modal)
pred_gg2_filt=focal(pred_gg2, w=matrix(1,3,3), fun=modal)
pred_prob_gg_1_filt=focal(pred_prob_gg_1, w=matrix(1,3,3), fun=modal)
pred_prob_gg_2_filt=focal(pred_prob_gg_2, w=matrix(1,3,3), fun=modal)

pred_gg_rast_filt=stack(pred_gg1_filt, pred_gg2_filt, pred_prob_gg_1_filt, pred_prob_gg_2_filt)
pred_gg_rast_filt=mask(pred_gg_rast_filt, mask_rast, maskvalue=1)
writeRaster(pred_gg_rast_filt, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_gg_filt.tif', format='GTiff')


#sg
#predict a sg for each point
variables_sg
stack_sg=stack(pred_gg1, Stand_Height, ari, MRVBF, Valley_Depth, MRRTF, Slope_Height, ndvi_med_oct, b2_fall)
stack_sg=rasterToPoints(stack_sg)
coords=stack_sg[,1:2]
pred_gg_values=stack_sg[,3]
stack_sg=stack_sg[,-c(1:3)]

colnames(stack_sg)=variables_sg[-1]

gg_values=data.frame(seq(1,length(table(pred_gg)), 1), names(table(pred_gg)))
colnames(gg_values)=c('code', 'great_group')
pred_gg_values=data.frame(pred_gg_values)
colnames(pred_gg_values)='code'
pred_gg_values=join(pred_gg_values, gg_values, by='code')

pred_gg_values=pred_gg_values$great_group


pred_sg1=data.frame(pred_sg1=as.character(), x=as.numeric(), y=as.numeric())
pred_sg2=data.frame(pred_sg1=as.character(), x=as.numeric(), y=as.numeric())
pred_prob_sg_1=data.frame(pred_sg1=as.character(), x=as.numeric(), y=as.numeric())
pred_prob_sg_2=data.frame(pred_sg1=as.character(), x=as.numeric(), y=as.numeric())
for (x in unique(pred_gg)){
  stack_sg_sub=stack_sg[pred_gg_values==x,]
  coords_sub=coords[pred_gg_values==x,]
  pred_sg_sub=predict(models_sg[[x]], stack_sg_sub)
  pred_sg=pred_sg_sub$predictions
  second_highest_sg = colnames(pred_sg)[apply(pred_sg, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
  pred_prob_sg=apply(pred_sg, 1, function(x) sort(x, decreasing=TRUE)[1])
  pred_prob_sg2=apply(pred_sg, 1, function(x) sort(x, decreasing=TRUE)[2])
  pred_sg=colnames(pred_sg)[max.col(pred_sg, ties.method="first")]
  pred_sg=data.frame(pred_sg, coords_sub)
  second_highest_sg=data.frame(second_highest_sg, coords_sub)
  pred_prob_sg1=data.frame(pred_prob_sg, coords_sub)
  pred_prob_sg2=data.frame(pred_prob_sg2, coords_sub)
  pred_sg1=rbind(pred_sg1, pred_sg)
  pred_sg2=rbind(pred_sg2, second_highest_sg)
  pred_prob_sg_1=rbind(pred_prob_sg_1, pred_prob_sg1)
  pred_prob_sg_2=rbind(pred_prob_sg_2, pred_prob_sg2)
}

pred_sg=pred_sg1$pred_sg
second_highest_sg=pred_sg2$second_highest_sg

pred_sg1$pred_sg=as.factor(pred_sg1$pred_sg)
pred_sg2$second_highest_sg=as.factor(pred_sg2$second_highest_sg)

coordinates(pred_sg1)=~x+y
coordinates(pred_sg2)=~x+y
coordinates(pred_prob_sg_1)=~x+y
coordinates(pred_prob_sg_2)=~x+y

pred_sg1=rasterFromXYZ(pred_sg1)
pred_sg2=rasterFromXYZ(pred_sg2)
pred_prob_sg_1=rasterFromXYZ(pred_prob_sg_1)
pred_prob_sg_2=rasterFromXYZ(pred_prob_sg_2)

pred_sg_rast=stack(pred_sg1, pred_sg2, pred_prob_sg_1, pred_prob_sg_2)
pred_sg_rast=mask(pred_sg_rast, mask_rast, maskvalue=1)
writeRaster(pred_sg_rast, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_sg.tif', format='GTiff')

write.csv(round(table(pred_sg)/sum(table(pred_sg))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/sg_1_table_2022_04_06.csv')
write.csv(round(table(second_highest_sg)/sum(table(second_highest_sg))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/sg_2_table_2022_04_06.csv')

plot(pred_sg1)
round(table(pred_sg)/sum(table(pred_sg))*100,2)[1]
round(table(second_highest_sg)/sum(table(second_highest_sg))*100,2)[3]

#median filter
pred_sg1_filt=focal(pred_sg1, w=matrix(1,3,3), fun=modal)
pred_sg2_filt=focal(pred_sg2, w=matrix(1,3,3), fun=modal)
pred_prob_sg_1_filt=focal(pred_prob_sg_1, w=matrix(1,3,3), fun=modal)
pred_prob_sg_2_filt=focal(pred_prob_sg_2, w=matrix(1,3,3), fun=modal)

pred_sg_rast_filt=stack(pred_sg1_filt, pred_sg2_filt, pred_prob_sg_1_filt, pred_prob_sg_2_filt)
pred_sg_rast_filt=mask(pred_sg_rast_filt, mask_rast, maskvalue=1)
writeRaster(pred_sg_rast_filt, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_sg_filt.tif', format='GTiff')


#point breakdown by training area
eia_sp=eia
coordinates(eia_sp)=~Easting+Northing

eia_sp=crop(eia_sp, roi)

table(eia_sp$Soil.Subgr)
table(eia_sp$grt.grp)
table(eia_sp$Order)

#unconstrained mapping
#gg
w <- 1/table(eia_gg$grt.grp)
w <- w^(1/4)
w <- w/sum(w)
weights <- rep(0, nrow(eia_gg))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_gg_sub$grt.grp==temp_label] <- w[i]
}
model_gg_figure_unconstrainted=ranger(as.factor(grt.grp)~., data=eia_gg[,-10], importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)

pred_gg_unconstrainted=predict(model_gg_figure_unconstrainted, stack_gg)
pred_gg_unconstrainted=pred_gg_unconstrainted$predictions
second_highest_gg_unconstrainted = colnames(pred_gg_unconstrainted)[apply(pred_gg_unconstrainted, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_prob_gg_unconstrainted=apply(pred_gg_unconstrainted, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_prob_gg_unconstrainted2=apply(pred_gg_unconstrainted, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_gg_unconstrainted=colnames(pred_gg_unconstrainted)[max.col(pred_gg_unconstrainted, ties.method="first")]

pred_gg_unconstrainted1=data.frame(as.factor(pred_gg_unconstrainted), coords)
pred_gg_unconstrainted2=data.frame(as.factor(second_highest_gg_unconstrainted), coords)
pred_prob_gg_unconstrainted_1=data.frame(pred_prob_gg_unconstrainted, coords)
pred_prob_gg_unconstrainted_2=data.frame(pred_prob_gg_unconstrainted2, coords)

coordinates(pred_gg_unconstrainted1)=~x+y
coordinates(pred_gg_unconstrainted2)=~x+y
coordinates(pred_prob_gg_unconstrainted_1)=~x+y
coordinates(pred_prob_gg_unconstrainted_2)=~x+y

pred_gg_unconstrainted1=rasterFromXYZ(pred_gg_unconstrainted1)
pred_gg_unconstrainted2=rasterFromXYZ(pred_gg_unconstrainted2)
pred_prob_gg_unconstrainted_1=rasterFromXYZ(pred_prob_gg_unconstrainted_1)
pred_prob_gg_unconstrainted_2=rasterFromXYZ(pred_prob_gg_unconstrainted_2)

pred_gg_unconstrainted_rast=stack(pred_gg_unconstrainted1, pred_gg_unconstrainted2, pred_prob_gg_unconstrainted_1, pred_prob_gg_unconstrainted_2)
pred_gg_unconstrainted_rast=mask(pred_gg_unconstrainted_rast, mask_rast, maskvalue=1)
writeRaster(pred_gg_unconstrainted_rast, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_gg_unconstrainted.tif', format='GTiff')

write.csv(round(table(pred_gg_unconstrainted)/sum(table(pred_gg_unconstrainted))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/gg_unconstrainted_1_table_2022_04_06.csv')
write.csv(round(table(second_highest_gg_unconstrainted)/sum(table(second_highest_gg_unconstrainted))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/gg_unconstrainted_2_table_2022_04_06.csv')

plot(pred_gg_unconstrainted1)
round(table(pred_gg_unconstrainted)/sum(table(pred_gg_unconstrainted))*100,2)[1]
round(table(second_highest_gg_unconstrainted)/sum(table(second_highest_gg_unconstrainted))*100,2)[4]

#modal filter
pred_gg_unconstrainted1_filt=focal(pred_gg_unconstrainted1, w=matrix(1,3,3), fun=modal)
pred_gg_unconstrainted2_filt=focal(pred_gg_unconstrainted2, w=matrix(1,3,3), fun=modal)
pred_prob_gg_unconstrainted_1_filt=focal(pred_prob_gg_unconstrainted_1, w=matrix(1,3,3), fun=modal)
pred_prob_gg_unconstrainted_2_filt=focal(pred_prob_gg_unconstrainted_2, w=matrix(1,3,3), fun=modal)

pred_gg_unconstrainted_rast_filt=stack(pred_gg_unconstrainted1_filt, pred_gg_unconstrainted2_filt, pred_prob_gg_unconstrainted_1_filt, pred_prob_gg_unconstrainted_2_filt)
pred_gg_unconstrainted_rast_filt=mask(pred_gg_unconstrainted_rast_filt, mask_rast, maskvalue=1)
writeRaster(pred_gg_unconstrainted_rast_filt, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_gg_unconstrainted_filt.tif', format='GTiff')




#sg
w <- 1/table(eia_order$Order)
w <- w/sum(w)
weights <- rep(0, nrow(eia_order))
for (i in 1:length(w)){
  temp_label=names(w)[i]
  weights[eia_order$Order==temp_label] <- w[i]
}

model_sg_figure_unconstrainted=ranger(as.factor(Soil.Subgr)~., data=eia_sg[,-1], importance='impurity',probability=TRUE, splitrule = "extratrees", case.weights = weights)

pred_sg_unconstrainted=predict(model_sg_figure_unconstrainted, stack_sg)
pred_sg_unconstrainted=pred_sg_unconstrainted$predictions
second_highest_sg_unconstrainted = colnames(pred_sg_unconstrainted)[apply(pred_sg_unconstrainted, 1, function(x)which(x != 0 & x == sort(x, decreasing = TRUE)[2])[1])]
pred_prob_sg_unconstrainted=apply(pred_sg_unconstrainted, 1, function(x) sort(x, decreasing=TRUE)[1])
pred_prob_sg_unconstrainted2=apply(pred_sg_unconstrainted, 1, function(x) sort(x, decreasing=TRUE)[2])
pred_sg_unconstrainted=colnames(pred_sg_unconstrainted)[max.col(pred_sg_unconstrainted, ties.method="first")]

pred_sg_unconstrainted1=data.frame(as.factor(pred_sg_unconstrainted), coords)
pred_sg_unconstrainted2=data.frame(as.factor(second_highest_sg_unconstrainted), coords)
pred_prob_sg_unconstrainted_1=data.frame(pred_prob_sg_unconstrainted, coords)
pred_prob_sg_unconstrainted_2=data.frame(pred_prob_sg_unconstrainted2, coords)

coordinates(pred_sg_unconstrainted1)=~x+y
coordinates(pred_sg_unconstrainted2)=~x+y
coordinates(pred_prob_sg_unconstrainted_1)=~x+y
coordinates(pred_prob_sg_unconstrainted_2)=~x+y

pred_sg_unconstrainted1=rasterFromXYZ(pred_sg_unconstrainted1)
pred_sg_unconstrainted2=rasterFromXYZ(pred_sg_unconstrainted2)
pred_prob_sg_unconstrainted_1=rasterFromXYZ(pred_prob_sg_unconstrainted_1)
pred_prob_sg_unconstrainted_2=rasterFromXYZ(pred_prob_sg_unconstrainted_2)

pred_sg_unconstrainted_rast=stack(pred_sg_unconstrainted1, pred_sg_unconstrainted2, pred_prob_sg_unconstrainted_1, pred_prob_sg_unconstrainted_2)
pred_sg_unconstrainted_rast=mask(pred_sg_unconstrainted_rast, mask_rast, maskvalue=1)
writeRaster(pred_sg_unconstrainted_rast, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_sg_unconstrainted.tif', format='GTiff')

write.csv(round(table(pred_sg_unconstrainted)/sum(table(pred_sg_unconstrainted))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/sg_unconstrainted_1_table_2022_04_06.csv')
write.csv(round(table(second_highest_sg_unconstrainted)/sum(table(second_highest_sg_unconstrainted))*100,2), '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/sg_unconstrainted_2_table_2022_04_06.csv')

plot(pred_sg_unconstrainted1)
round(table(pred_sg_unconstrainted)/sum(table(pred_sg_unconstrainted))*100,2)[1]
round(table(second_highest_sg_unconstrainted)/sum(table(second_highest_sg_unconstrainted))*100,2)[4]

#modal filter
pred_sg_unconstrainted1_filt=focal(pred_sg_unconstrainted1, w=matrix(1,3,3), fun=modal)
pred_sg_unconstrainted2_filt=focal(pred_sg_unconstrainted2, w=matrix(1,3,3), fun=modal)
pred_prob_sg_unconstrainted_1_filt=focal(pred_prob_sg_unconstrainted_1, w=matrix(1,3,3), fun=modal)
pred_prob_sg_unconstrainted_2_filt=focal(pred_prob_sg_unconstrainted_2, w=matrix(1,3,3), fun=modal)

pred_sg_unconstrainted_rast_filt=stack(pred_sg_unconstrainted1_filt, pred_sg_unconstrainted2_filt, pred_prob_sg_unconstrainted_1_filt, pred_prob_sg_unconstrainted_2_filt)
pred_sg_unconstrainted_rast_filt=mask(pred_sg_unconstrainted_rast_filt, mask_rast, maskvalue=1)
writeRaster(pred_sg_unconstrainted_rast_filt, '/home/preston/OneDrive/Papers/Soil_remote_sensing/Figures/Hierarchical/pred_sg_unconstrainted_filt.tif', format='GTiff')


