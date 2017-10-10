import org.bitcoinj.core.Block;
import org.bitcoinj.core.Context;
import org.bitcoinj.core.NetworkParameters;
import org.bitcoinj.core.Transaction;
import org.bitcoinj.params.MainNetParams;
import org.bitcoinj.utils.BlockFileLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cxa123230 on 10/6/2017.
 */

public class ChainParser {
    public static void main(String [] args) throws IOException {

        String dir = "D:\\Bitcoin\\blocks\\";
        Context.getOrCreate(MainNetParams.get());
        NetworkParameters np = new MainNetParams();
        int j=0;
        BufferedWriter wr = new BufferedWriter(new FileWriter("D:\\Bitcoin\\createddata\\txSums.txt"));//txall
        for (int i = 0; i < 5000; i++) {
            String fName ="";
            if(i<10)  fName = "0000"; else if(i<100) fName ="000"; else fName = "00";
            String pathname = dir + "blk" + fName+i+".dat";
            if(new File(pathname).exists()){
                System.out.println(pathname+ "   added.");
                List<File> blockChainFiles = new ArrayList<>();
                blockChainFiles.add(new File(pathname));
                BlockFileLoader bfl = new BlockFileLoader(np, blockChainFiles);


                StringBuffer bf = new StringBuffer();
                for (Block block : bfl) {
                    if(++j%10000==0) System.out.println(j);
                    for(Transaction tx :block.getTransactions()){
//                        int inputs = tx.getInputs().size();
//                        int outputs = tx.getOutputs().size();
                        long sum = tx.getOutputSum().getValue();

                        bf.append(tx.getHashAsString() + "\t" + block.getTimeSeconds() + "\t" + sum + "\r\n");
                        //bf.append(tx.getHashAsString()+"\t"+block.getTimeSeconds()+"\t"+sum+"\t"+inputs+"\t"+outputs+"\r\n");

                    }
                }
                wr.write(bf.toString());
            }
        }


        wr.close();

    }

}
