
#install.packages(c("plyr", "ggplot2", "ineq"))
require(plyr)
require(ggplot2)
require(ineq)


rm(list=ls(all=TRUE))
readFile<-function(fileN){
  results <- read.csv(paste("D://bitcoin/createddata/results/",fileN,sep=""),sep="\t",header=F)
  return(results)
}
 

fileDL = "occslidingPrediction_priceless.csv"#,"WithoutChainletslidingPrediction.csv"
resDL <- readFile(fileDL)
colnames(resDL) <- c("priced","aggregated","window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day")
dataDL<- ddply(resDL,.(slideTrain,slideTest,window,horizon), summarize,
                  meanDev = mean(abs(realLR-predictedLR)))
dataDL$method<-"DL"

fileB = "betti_prediction_priceless.csv"
resBettiPriceless <-readFile(fileB)
colnames(resBettiPriceless) <- c("allowed", "window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day") 
dataBetti<- ddply(resBettiPriceless,.(slideTrain,slideTest,window,horizon), summarize,
              meanDev = mean(abs(realLR-predictedLR)))
dataBetti$method<-"bettiPriceless" 
  
 
fileBP = "betti_prediction_with_price.csv"
resBettiPriced <-readFile(fileBP)
colnames(resBettiPriced) <- c("allowed", "window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day") 
dataBettiP<- ddply(resBettiPriced,.(slideTrain,slideTest,window,horizon), summarize,
                  meanDev = mean(abs(realLR-predictedLR)))
dataBettiP$method<-"bettiPriced" 


aggPDL<-nrow(resDL[resDL$realLR>0&resDL$predictedLR>0,])
aggNDL<-nrow(resDL[resDL$realLR<=0&resDL$predictedLR<=0,])
aggN<-nrow(resDL)

aggPBettiPriceless<-nrow(resBettiPriceless[resBettiPriceless$realLR>0&resBettiPriceless$predictedLR>0,])
aggNBettiPriceless<-nrow(resBettiPriceless[resBettiPriceless$realLR<=0&resBettiPriceless$predictedLR<=0,])
aggB<-nrow(resBettiPriceless)

aggPBPriced<-nrow(resBettiPriced[resBettiPriced$realLR>0&resBettiPriced$predictedLR>0,])
aggNBPriced<-nrow(resBettiPriced[resBettiPriced$realLR<=0&resBettiPriced$predictedLR<=0,])
aggBP<-nrow(resBettiPriced)
aggResults<-(c(aggPDL/aggN,aggNDL/aggN,aggN,"DL"))
aggResults<-rbind(aggResults,c(aggPBettiPriceless/aggB,aggNBettiPriceless/aggB,aggB,"betti priceless"),c(aggPBPriced/aggBP,aggNBPriced/aggBP,aggBP,"Betti priced"))
colnames(aggResults)<-c("Positive","Negative","Total","Method")
View(rbind(dataDL,dataBetti,dataBettiP) )
View(aggResults)
par(mfrow=c(1,4))
plot(resBettiPriceless$predictedLR,resBettiPriceless$realLR)
plot(resBettiPriced$predictedLR,resBettiPriced$realLR)
plot(resDL$predictedLR,resDL$realLR)

