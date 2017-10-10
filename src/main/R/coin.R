require(plyr)
require(ggplot2)
require(ineq)

rm(list=ls(all=TRUE))


 

 
wdata <- read.csv("C:/Users/cxa123230/IdeaProjects/CoinWorks/src/main/resources/transactions.txt",sep="\t",header=T)
wdata$axis<-paste(wdata$year,"/",wdata$day) 
  
 
m13<- ggplot(wdata,aes(x=paste(year,day),y=split))+geom_line(size=2)+scale_x_continuous(name="day")+theme_bw()+theme(legend.position = c(0.4, 0.8),text = element_text(size=16))+scale_y_continuous(limits=c(0.05,0.27),name="Gini value")
m13<-m13+geom_point(aes(y=merge), size=4) +geom_point(aes(y=transition), size=4)   
m13
ggsave(filename=paste(resultFolder,"figs/giniUseless.eps",sep=""),plot=m13,width=5,height=3,unit="in")
ggsave(filename=paste(resultFolder,"figs/giniUseless.png",sep=""),plot=m13,width=5,height=3,unit="in")
