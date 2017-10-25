import org.bitcoinj.core.*;
import org.bitcoinj.params.MainNetParams;
import org.bitcoinj.utils.BlockFileLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by cxa123230 on 10/6/2017.
 */

public class BackwardBranchExtractor {
    static NetworkParameters np = new MainNetParams();

    public static void main(String[] args) throws IOException {

        String dir = "D:\\Bitcoin\\blocks\\";
        Context.getOrCreate(MainNetParams.get());

        HashSet<String> addresses = new HashSet<String>() {
        };
        int maxWave = 3;
        String d2 = "D:/Bitcoin/createddata/subchain/";
        String typeofData = "blackmail";
        BufferedReader fr = new BufferedReader(new FileReader(d2 + typeofData + "addresses.txt"));
        String line = "";
        HashMap<String, Integer> waves = new HashMap<String, Integer>();
        while ((line = fr.readLine()) != null) {
            addresses.add(line.trim());
            waves.put(line.trim(), 0);
        }
        fr.close();
        System.out.println(addresses.size() + "  addresses added.");


        int j = 0;
        int wave = 0;
        String fileName = d2 + typeofData + "transactions.txt";
        BufferedWriter wr = new BufferedWriter(new FileWriter(fileName));


        System.out.println(wave + " wave.");
        for (int i = 5000; i > 0; i--) {
            String pathname = getFileString(dir, i);
            if (new File(pathname).exists()) {
                System.out.println(pathname + "   file.");
                List<File> blockChainFiles = new ArrayList<>();
                blockChainFiles.add(new File(pathname));
                BlockFileLoader bfl = new BlockFileLoader(np, blockChainFiles);

                for (Block block : bfl) {
                    if (++j % 100000 == 0) System.out.println(j);
                    for (Transaction tx : block.getTransactions()) {
                        String txId = tx.getHashAsString();


                        for (TransactionOutput y : tx.getOutputs()) {
                            try {
                                Address outputAddr = y.getAddressFromP2SH(np);
                                if (outputAddr == null) {
                                    outputAddr = y.getAddressFromP2PKHScript(np);
                                    if (outputAddr == null) {
                                        //System.out.println("unparseable output addr ");
                                        continue;
                                    }
                                }
                                String outAddStr = outputAddr.toString();
                                if (addresses.contains(outAddStr)) {
                                    int waveOfInAdd = waves.get(outAddStr) + 1;
                                    if (waveOfInAdd <= maxWave) {

                                        for (TransactionInput input : tx.getInputs()) {
                                            String inAdd = input.getFromAddress().toString();
                                            addresses.add(inAdd);
                                            waves.put(inAdd, waveOfInAdd);
//                                            System.out.println(txId + " adding " + inAdd+" "+waves.get(inAdd));
                                        }
                                        writetx(tx, wr, waveOfInAdd);
                                        break;
                                    }


                                }


                            } catch (ScriptException e) {
                                System.out.println("Error in parsing output");
                            }
                        }
                    }
                }
            }
        }


        wr.close();

    }

    private static String getFileString(String dir, int i) {
        String fName = "";
        if (i < 10) fName = "0000";
        else if (i < 100) fName = "000";
        else fName = "00";
        return dir + "blk" + fName + i + ".dat";
    }

    private static void writetx(Transaction tx, BufferedWriter wr, int wave) throws IOException {

        {
            String id = tx.getHashAsString();
            for (TransactionInput input : tx.getInputs()) {
                String prAddress = input.getFromAddress().toString();
                wr.write("i\t" + wave + "\t" + id + "\t" + prAddress + "\r\n");
            }

            for (TransactionOutput y : tx.getOutputs()) {
                try {
                    long coin = y.getValue().getValue();
                    Address outputAddr = y.getAddressFromP2SH(np);
                    if (outputAddr == null) {
                        outputAddr = y.getAddressFromP2PKHScript(np);
                        if (outputAddr == null) {
                            //System.out.println("unparseable output addr ");
                            continue;
                        }
                    }
                    wr.write("o\t" + wave + "\t" + id + "\t" + outputAddr.toString() + "\r\n");
                } catch (ScriptException e) {
                    System.out.println("Error in parsing output");
                }
            }

        }
        wr.flush();
    }

}
