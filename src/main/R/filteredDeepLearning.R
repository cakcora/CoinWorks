
#install.packages(c("plyr", "ggplot2", "ineq"))
require(plyr)
require(ggplot2)


rm(list=ls(all=TRUE))
readFile<-function(fileN){
  results <- read.csv(paste("D://bitcoin/createddata/results/",fileN,sep=""),sep="\t",header=F)
  return(results)
}

confusion<-function(results){
  aggB<-nrow(results)
  tp<-nrow(results[results$realLR>0&results$predictedLR>0,])/aggB
  tn<-nrow(results[results$realLR<=0&results$predictedLR<=0,])/aggB
  fn<- nrow(results[results$realLR>0&results$predictedLR<=0,])/aggB
  fp<-nrow(results[results$realLR<=0&results$predictedLR>0,])/aggB
  return(c(tp,fp,tn,fn,tp+tn))
}


fileDL = "occboostingPrediction.csv" 
resFL <- readFile(fileDL)
colnames(resFL) <- c("priced","feature","window","horizon","threshold","slideTrain","slideTest","predictedLR","realLR","year","day")
dat<- ddply(resFL,.(threshold,priced,feature,slideTrain,slideTest,window,horizon), summarize,
            rmse = mean((realLR-predictedLR)^2), days=length(realLR))
dat$rmse<-dat$rmse^0.5
z<-ddply(resFL, .(threshold,priced,feature,slideTrain,slideTest,window,horizon), function(x) confusion(x))
colnames(z) <- c("threshold","priced","feature","slideTrain","slideTest","window","horizon","tp","fp","tn","fn","acc")
aggFL<-cbind(z,dat$rmse,dat$days)
View(aggFL)



