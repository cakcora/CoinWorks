#---------------------------EXP 1-------------------------

#install.packages(c("plyr", "ggplot2", "ineq"))
require(plyr)
require(ggplot2)

rm(list=ls(all=TRUE))
dir<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Deep learning/results/result/log_return/changing_betti_filtration"
bettiFile<<-paste(dir,"/res.csv",sep="")

readFile<-function(fileN){
  results <- read.csv(paste(fileN,sep=""),sep="\t",header=T)
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

 
 
 
 
results <- readFile(bettiFile)
plt1<- ggplot(results)+geom_line(aes(x=filtration,y=rmse))+theme_classic()+theme(text = element_text(size=16))+scale_x_continuous(breaks=c(50,100,200,400))
plt1
resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Deep learning/fig/"
ggsave(filename=paste(resultFolder,"bettiresults.eps",sep=""),plot=plt1,width=5,height=3,unit="in")

##-----------------------------------------------------------------

resDL <- readFile(fileDL)
colnames(resDL) <- c("priced","feature","window","horizon","slideTrain","slideTest","predictedLR","realLR","day")
maxDay<-max(resDL$horizon)
#resDL<-resDL[resDL$day<=(365-maxDay),]
dat1<- ddply(resDL,.(priced,feature,slideTrain,window,horizon), summarize,
             rmse = mean((realLR-predictedLR)^2), days=length(realLR))
dat1$rmse<-dat1$rmse^0.5
dat1$method<-"Price only"
z<-ddply(resDL, .(priced,feature,slideTrain,window,horizon), function(x) confusion(x))
colnames(z) <- c("priced","feature","slideTrain","window","horizon","tp","fp","tn","fn","acc")
aggDL<-cbind(z,method=dat1$method,rmse=dat1$rmse,days=dat1$days)



fileB = "betti_prediction.csv"
resBetti <-readFile(fileB)
colnames(resBetti) <- c("priced","feature", "window","horizon","slideTrain","slideTest","predictedLR","realLR","year","day") 
maxDay<-max(resBetti$horizon)
#resBetti<-resBetti[resBetti$day<=(365-maxDay),]
dat2<- ddply(resBetti,.(priced,feature,slideTrain,slideTest,window,horizon), summarize,
             rmse = mean(abs(realLR-predictedLR)^2), days=length(realLR))
dat2$rmse<-dat2$rmse^0.5
dat2$method<-"Betti" 
z2<-ddply(resBetti, .(priced,feature,slideTrain,window,horizon), function(x) confusion(x))

colnames(z2) <- c("priced","feature","slideTrain","window","horizon","tp","fp","tn","fn","acc")
aggBetti<-cbind(z2,method=dat2$method,rmse=dat2$rmse,days=dat2$days)


exp2<-( rbind(aggDL,aggBetti) )
View(exp2)
plt2<-ggplot(data=exp2, aes(x=horizon,y=rmse)) +
  geom_point(na.rm = TRUE,size=2)+
  theme_classic()+theme(text = element_text(size=16))+
  scale_x_continuous(name= "x axis")+
  scale_y_continuous(name="Rmse")
plt2
#ggsave(filename=paste("C:/Users/cakcora/Dropbox/Publications/PostDoc work/Deep learning/fig/exp2.pdf",sep=""),plot=plt2,width=5,height=3,unit="in")

wind<-3
hori<-1
mday<-300
ggplot(resBetti[resBetti$window==wind&resBetti$horizon==hori&resBetti$day>mday,])+
  geom_line(aes(y=realLR,x=day),color="black")+
  geom_line(aes(y=predictedLR,x=day),color="green")
#geom_line(aes(y=resDL[resDL$window==wind&resDL$horizon==hori&resDL$day>mday,]$predictedLR,x=resDL[resDL$window==wind&resDL$horizon==hori&resDL$day>mday,]$day),color="red")
#ggplot(resDL[resDL$window==wind&resDL$horizon==hori&resDL$day>mday,])+geom_line(aes(y=realLR,x=day),color="green")+geom_line(aes(y=predictedLR,x=day))
##########################################################
if(FALSE){
  fff<<-"C:/Users/cakcora/Dropbox/Publications/PostDoc work/Deep learning/results/results_for_cuneyt/occboostingPrediction.csv"
  results <- read.csv(fff,sep="\t",header=F)
  colnames(results) <- c("priced","feature", "window","horizon","threshold","slideTrain","slideTest","predictedLR","realLR","year","day") 
  ddd<- ddply(results[results$realLR<0.1,],.(priced,feature,slideTrain,slideTest,window,horizon,threshold), summarize,
              rmse = mean(abs(realLR-predictedLR)^2), days=length(realLR))
  ddd$rmse<-ddd$rmse^0.5
  ddd$method<-"FL" 
  View(ddd[ddd$window==3&ddd$horizon==1,])
}