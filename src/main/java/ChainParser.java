import org.bitcoinj.core.*;
import org.bitcoinj.params.MainNetParams;
import org.bitcoinj.utils.BlockFileLoader;

/**
 * Created by cxa123230 on 10/6/2017.
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ChainParser {
    public static void main(String [] args) throws IOException {
        String dir = "D:\\Bitcoin\\blocks\\";
        Context.getOrCreate(MainNetParams.get());
        NetworkParameters np = new MainNetParams();
        int j=0;
        BufferedWriter wr = new BufferedWriter(new FileWriter("D:\\Bitcoin\\createddata\\txall.txt"));
        for(int i=0;i<1000;i++){
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
                        int inputs = tx.getInputs().size();
                        int outputs = tx.getOutputs().size();

                        bf.append(tx.getHashAsString()+"\t"+block.getTimeSeconds()+"\t"+inputs+"\t"+outputs+"\r\n");

                    }
                }
                wr.write(bf.toString());
            }
        }


        wr.close();

    }

}
