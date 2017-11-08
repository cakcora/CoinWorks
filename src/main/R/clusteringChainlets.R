require(plyr)
require(ggplot2)
require(ineq)
require(TSclust)

rm(list=ls(all=TRUE))
 
resultFolder<-"C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/R codes and Figures/"

chainlets <- read.csv("C:/Users/cxa123230/Dropbox/Publications/PostDoc work/Blockchain Survey/Blockchain Graphlets/Data/bitcoin/pricedBitcoin.csv",sep="\t",header=T)
chainlets<-chainlets[chainlets$year>2015,]

ch <- cbind(chainlets[,5:9])
ch <- cbind(ch,chainlets[,25:29])
ch <- cbind(ch,chainlets[,45:49])
ch <- cbind(ch,chainlets[,65:69])
ch <- cbind(ch,chainlets[,85:89])
#ch <- cbind(ch,chainlets[,260:264])

deucl <- diss(ch, "FRECHET") 
#deucl <- diss(chainlets[,5:404], "DWT") 
hceucl <- hclust(deucl, "complete")
m13<-plot(hceucl)


#ggsave(filename=paste(resultFolder,"dend.pdf",sep=""),plot=m13,width=5,height=3,unit="in")
