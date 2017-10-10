require(plyr)
require(ggplot2)
require(ineq)

rm(list=ls(all=TRUE))


 

 
wdata <- read.csv("C:/Users/cxa123230/IdeaProjects/CoinWorks/src/main/resources/trWeekly.txt",sep="\t",header=T)
wdata$myx<-paste(wdata$year,"/",wdata$week) 
wdata$ax<-as.factor(wdata$ax)  
dfm <- melt(as.data.frame(wdata), id.vars=c("myx", "year","week"))

m13<-ggplot(data = dfm, aes(x = myx,y=value,group=variable,colour=variable)) + 
  geom_line() +scale_x_discrete(name="week")+scale_y_continuous(name="Percentage")+theme_bw()+theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position = c(0.8, 0.6),text = element_text(size=16))
    

resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
ggsave(filename=paste(resultFolder,"transactiontypes.eps",sep=""),plot=m13,width=5,height=3,unit="in")
ggsave(filename=paste(resultFolder,"transactiontypes.png",sep=""),plot=m13,width=5,height=3,unit="in")
