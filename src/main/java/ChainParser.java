import org.bitcoinj.core.*;
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

        String blockDir = "D:\\Bitcoin\\blocks\\";
        Context.getOrCreate(MainNetParams.get());
        NetworkParameters np = new MainNetParams();
        int j=0;
        String inputDir = "D:\\Bitcoin\\createddata\\";
        BufferedWriter wrO = new BufferedWriter(new FileWriter(inputDir + "txOutputSums.txt"));
        BufferedWriter wrI = new BufferedWriter(new FileWriter(inputDir + "txinputs.txt"));
        BufferedWriter wrID = new BufferedWriter(new FileWriter(inputDir + "txIDs.txt"));
        for (int i = 0; i < 5000; i++) {
            String fName ="";
            if(i<10)  fName = "0000"; else if(i<100) fName ="000"; else fName = "00";
            String pathname = blockDir + "blk" + fName + i + ".dat";
            if(new File(pathname).exists()){
                System.out.println(pathname+ "   added.");
                List<File> blockChainFiles = new ArrayList<>();
                blockChainFiles.add(new File(pathname));
                BlockFileLoader bfl = new BlockFileLoader(np, blockChainFiles);


                for (Block block : bfl) {
                    StringBuffer oBuffer = new StringBuffer();
                    StringBuffer iBuffer = new StringBuffer();
                    if(++j%10000==0) System.out.println(j);
                    for(Transaction tx :block.getTransactions()){

                        try {
                            List<TransactionInput> inputs = tx.getInputs();
                            List<TransactionOutput> outputs = tx.getOutputs();
                            if (!tx.isCoinBase()) {
                                iBuffer.append(block.getTimeSeconds() + "\t" + tx.getHashAsString());
                                for (TransactionInput ti : inputs) {
                                    TransactionOutPoint outpoint = ti.getOutpoint();
                                    long parentOutputIndex = outpoint.getIndex();
                                    iBuffer.append("\t" + outpoint.getHash().toString() + "\t" + parentOutputIndex);
                                }
                                iBuffer.append("\r\n");
                                oBuffer.append(block.getTimeSeconds() + "\t" + tx.getHashAsString());
                                for (TransactionOutput to : outputs) {

                                    Address ts = to.getAddressFromP2PKHScript(np);
                                    Address tp = to.getAddressFromP2SH(np);

                                    Coin value = to.getValue();
                                    if (ts != null) {
                                        oBuffer.append("\t" + ts + "\t" + value);
                                    } else if (tp != null) {
                                        oBuffer.append("\t" + tp + "\t" + value);
                                    } else oBuffer.append("\t" + "noaddress" + "\t" + value);
                                }
                                oBuffer.append("\r\n");

                            }
                        } catch (ScriptException e) {
                            e.printStackTrace();
                            oBuffer.append("\r\n");
                            iBuffer.append("\r\n");
                        }
                    }
                    wrO.write(oBuffer.toString());
                    wrI.write(iBuffer.toString());
                }
            }
        }
        wrO.close();
        wrI.close();

    }



}
