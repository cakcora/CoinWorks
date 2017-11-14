require(plyr)
require(ggplot2)
require(ineq)
 
rm(list=ls(all=TRUE))
 
 
data <- read.csv("C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/Blockchain Graphlets/Random_Forest_Models_V3_All30.txt",sep="\t",header=T)
data<-data[data$Model!="Model0",] 
 

 
m13<-ggplot(data = data, aes(x = h,y=rRMSE,group = Model, color =Model)) + geom_line(size=1)+theme_bw()
m13<-m13+ scale_x_continuous(name="Prediction in Days",breaks=c(1,5,10,15,20,25,30))
m13<-m13+ scale_y_continuous(name="rRMSE")
m13<-m13+theme(axis.text.x = element_text(size = 10),legend.position = c(0.8, 0.25),axis.text.y = element_text(size=10))
 
m13    


resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
#ggsave(filename=paste(resultFolder,"7modelPerformance.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
#ggsave(filename=paste(resultFolder,"modelPerformance.png",sep=""),plot=m13,width=5,height=3,unit="in")


 