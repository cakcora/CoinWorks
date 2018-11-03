package Price;

import org.bitcoinj.core.*;
import org.bitcoinj.params.MainNetParams;
import org.bitcoinj.utils.BlockFileLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private static final Logger log = LoggerFactory.getLogger(ChainParser.class);

    public static void main(String [] args) throws IOException {

        String blockDir = args[0];
        Context.getOrCreate(MainNetParams.get());
        NetworkParameters np = new MainNetParams();
        int j=0;
        log.info(args[0]);
        String inputDir = args[1];
        String newInput = args[2];
        BufferedWriter fileWriter = new BufferedWriter(new FileWriter(inputDir + newInput));
        for (int i = 25; i < 5000; i++) {
            String fName ="";
            if (i < 10) fName = "0000";
            else if (i < 100) fName = "000";
            else if (i < 1000) fName = "00";
            else fName = "0";
            String pathname = blockDir + "blk" + fName + i + ".dat";
            if (new File(pathname).exists()) {
                System.out.println(pathname + "   added.");
                List<File> blockChainFiles = new ArrayList<>();
                blockChainFiles.add(new File(pathname));
                BlockFileLoader bfl = new BlockFileLoader(np, blockChainFiles);

                try {
                    for (Block block : bfl) {
                        StringBuffer oBuffer = new StringBuffer();
                        StringBuffer iBuffer = new StringBuffer();
                        if (++j % 10000 == 0) System.out.println(j);
                        System.out.println(block.toString());
                        for (Transaction tx : block.getTransactions()) {
                            try {
                                if (!tx.isCoinBase()) {

                                    iBuffer.append(block.getTimeSeconds() + "\t" + tx.getHashAsString());
                                    for (long in = 0; in < tx.getInputs().size(); in++) {
                                        TransactionInput ti = tx.getInput(in);

                                        TransactionOutPoint outpoint = ti.getOutpoint();

                                        long parentOutputIndex = outpoint.getIndex();
                                        iBuffer.append("\t" + outpoint.getHash().toString() + "\t" + parentOutputIndex);
                                    }
                                    iBuffer.append("\r\n");
                                    oBuffer.append(block.getTimeSeconds() + "\t" + tx.getHashAsString());
                                    for (long on = 0; on < tx.getOutputs().size(); on++) {
                                        TransactionOutput to = tx.getOutput(on);
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
                        fileWriter.write(iBuffer.toString());
                        fileWriter.flush();
                    }
                } catch (Exception re) {
                    System.out.println(" Failed block.");
                    //re.printStackTrace();

                }
            }
        }
        fileWriter.close();

    }


}
