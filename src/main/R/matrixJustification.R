require(plyr)
require(ggplot2)
require(ineq)

rm(list=ls(all=TRUE))
 
 
chainlets <- read.csv("C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/Blockchain Graphlets/Data/bitcoin/pricedBitcoin.csv",sep="\t",header=T)
#chainlets<-chainlets[chainlets$year>2011,]
#chainlets<-chainlets[chainlets$day%%7==1,]
chainlets$myx<-paste(chainlets$year,chainlets$day,sep="/ ") 

maxDim <-20 

matrixChains<-c("totaltx","myx")
for(x in 1:maxDim){ 
  for(y in 1:maxDim){ 
    matrixChains<-c(matrixChains,paste("X",x,".",y,sep=""))
  }
}


 
d<- subset(chainlets, select=matrixChains)
d$mSum <-rowSums(d[,4:length(d)]) 
 

br<-c("2011/ 152","2015/ 168","2017/ 160")
m13<-ggplot(data = d, aes(x = myx,y=mSum/totaltx,group = 1)) + geom_line()+theme_classic()+scale_x_discrete(name="Day",breaks=br,
                  labels=br)+scale_y_continuous(name="Percentage of chainlets")+theme(axis.text.x = element_text(size = 10),legend.position = c(0.8, 0.85),axis.text.y = element_text(size=10))
m13    


resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"
#ggsave(filename=paste(resultFolder,"matrixJustification.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
#ggsave(filename=paste(resultFolder,"matrixJustification.png",sep=""),plot=m13,width=5,height=3,unit="in")

res<-data.frame()
for(mxChain in 2:19){
  chChains <- c("totaltx")
  for(chX in 1:mxChain){
    for(chY in 1:mxChain){
      chChains<-c(chChains,paste("X",chX,".",chY,sep=""))
    }
  }
   
  chD<- subset(chainlets, select=chChains)
  chD$mSum <-rowSums(chD[,2:length(chD)]) 
  res<-rbind(res, c(mxChain,median(chD$mSum)/median(chD$totaltx)))
}
print(res)
 plot(res)