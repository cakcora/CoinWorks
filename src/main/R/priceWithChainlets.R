require(plyr)
require(ggplot2)
require(ineq)

rm(list=ls(all=TRUE))
 
 
chainlets <- read.csv("C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/Blockchain Graphlets/Data/bitcoin/pricedBitcoin.csv",sep="\t",header=T)
chainlets<-chainlets[chainlets$year>2014,]
chainlets<-chainlets[chainlets$day%%7==1,]

chainlets$myx<-paste(chainlets$year,"/",chainlets$day) 

splitChains<-c("myx", "price","totaltx","X1.2","X1.3","X1.4")

 
d<- subset(chainlets, select=splitChains)

d$price<-d$price/max(d$price)

d$X1.2<-d$X1.2/max(d$X1.2)
d$X1.3<-d$X1.3/max(d$X1.3)
d$X1.4<-d$X1.4/max(d$X1.4)
keeps<-c(1,4,6,7)   
dfm <- melt(as.data.frame(d), id.vars=c("myx","totaltx"))
attach(dfm)
m13<-ggplot(data = dfm, aes(x = myx,y=value,group=variable,colour=variable)) + 
  geom_line() +scale_x_discrete(name="day",breaks=levels(dfm$myx)[keeps],
                                                              labels=table(dfm$myx)[keeps])
  scale_y_continuous(name="Percentage")+theme_bw()+theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position = c(0.8, 0.85),text = element_text(size=10))
m13    

resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
ggsave(filename=paste(resultFolder,"chainletSplit.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
ggsave(filename=paste(resultFolder,"chainletSplit.png",sep=""),plot=m13,width=5,height=3,unit="in")


mergeChains<-c("myx", "price","totaltx","X2.1","X3.1","X4.1")
d2<- subset(chainlets, select=mergeChains)

d2$price<-d2$price/max(d2$price)

d2$X2.1<-d2$X2.1/max(d2$X2.1)
d2$X3.1<-d2$X3.1/max(d2$X3.1)
d2$X4.1<-d2$X4.1/max(d2$X4.1)
   
dfm2 <- melt(as.data.frame(d2), id.vars=c("myx","totaltx"))
attach(dfm2)
m13<-ggplot(data = dfm2, aes(x = myx,y=value,group=variable,colour=variable)) + 
  geom_line() +scale_x_discrete(name="day",breaks=levels(dfm$myx)[keeps],
                                labels=table(dfm$myx)[keeps])
scale_y_continuous(name="Percentage")+theme_bw()+theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position = c(0.8, 0.85),text = element_text(size=10))
m13    

resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
ggsave(filename=paste(resultFolder,"chainlettypesMerge.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
ggsave(filename=paste(resultFolder,"chainlettypesMerge.png",sep=""),plot=m13,width=5,height=3,unit="in")

extremeChains<-c("myx", "price","totaltx","X20.20","X20.19","X19.20")
dextreme<- subset(chainlets, select=extremeChains)

dextreme$price<-dextreme$price/max(dextreme$price)

dextreme$X20.20<-dextreme$X20.20/max(dextreme$X20.20)
dextreme$X20.19<-dextreme$X20.19/max(dextreme$X20.19)
dextreme$X19.20<-dextreme$X19.20/max(dextreme$X19.20)
   
dfmextreme <- melt(as.data.frame(dextreme), id.vars=c("myx","totaltx"))
attach(dfm2)
m13<-ggplot(data = dfmextreme, aes(x = myx,y=value,group=variable,colour=variable)) + 
  geom_line() +scale_x_discrete(name="day",breaks=levels(dfm$myx)[keeps],
                                labels=table(dfm$myx)[keeps])
scale_y_continuous(name="Percentage")+theme_bw()+theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position = c(0.8, 0.85),text = element_text(size=10))
m13    

resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
ggsave(filename=paste(resultFolder,"chainlettypesExtreme.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
ggsave(filename=paste(resultFolder,"chainlettypesExtreme.png",sep=""),plot=m13,width=5,height=3,unit="in")

