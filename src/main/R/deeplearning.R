
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
 

fileDL = "occslidingPrediction.csv" 
resDL <- readFile(fileDL)
colnames(resDL) <- c("priced","feature","window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day")
dat<- ddply(resDL,.(priced,feature,slideTrain,slideTest,window,horizon), summarize,
                  rmse = mean((realLR-predictedLR)^2), days=length(realLR))
dat$rmse<-dat$rmse^0.5
dat$method<-"chainlets"
z<-ddply(resDL, .(priced,feature,slideTrain,slideTest,window,horizon), function(x) confusion(x))
colnames(z) <- c("priced","feature","slideTrain","slideTest","window","horizon","tp","fp","tn","fn","acc")
aggDL<-cbind(z,dat$method,dat$rmse,dat$days)
 


fileB = "betti_prediction.csv"
resBetti <-readFile(fileB)
colnames(resBetti) <- c("priced","feature", "window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day") 
dat<- ddply(resBetti,.(priced,feature,slideTrain,slideTest,window,horizon), summarize,
              rmse = mean(abs(realLR-predictedLR)^2), days=length(realLR))
dat$rmse<-dat$rmse^0.5
dat$method<-"betti" 
z<-ddply(resBetti, .(priced,feature,slideTrain,slideTest,window,horizon), function(x) confusion(x))

colnames(z) <- c("priced","feature","slideTrain","slideTest","window","horizon","tp","fp","tn","fn","acc")
aggBetti<-cbind(z,dat$method,dat$rmse,dat$days)
 
 
View( rbind(aggDL,aggBetti) )
 
