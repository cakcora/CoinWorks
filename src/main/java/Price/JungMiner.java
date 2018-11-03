package Price;

import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.SparseGraph;
import org.joda.time.DateTime;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by cxa123230 on 10/2/2017.
 * takes txinall and txoutall files from the output of GraphParser and finds graphlets per the used time period (e.g., week)
 */


public class JungMiner {
    public static void main(String [] args) throws Exception {
        String dir = "C:\\Projects\\Coin\\bitcoin_dataset\\";
        String crdir = dir+"createddata\\";
        for (int year = 2011; year < 2015; year++) {
            for (int week = 1; week <= 52; week++) {
//                int year=2011;
//                int month = 11;
                int dim = 20;
                int [][] occM = new int[dim][dim];
                int[][] sumM = new int[dim][dim];
                Graph<Long, Integer> ig = new SparseGraph<Long, Integer>();
                BufferedReader inBr = new BufferedReader(new FileReader(dir + "txInAll.txt"));
                BufferedReader outBr = new BufferedReader(new FileReader(dir + "txOutAll.txt"));
                String line = "";
                HashMap<Long, HashSet<Long>> inTranx = new HashMap<Long, HashSet<Long>>();
                HashMap<Long, HashSet<Long>> outTranx = new HashMap<Long, HashSet<Long>>();
                HashMap<Long,  Long > sums = new HashMap<Long,  Long >();
                loadTranx(inBr, year, week, inTranx,sums);
                loadTranx(outBr, year, week, outTranx,sums);
                int sIn = inTranx.size();
                int sOut = outTranx.size();
                outTranx.keySet().retainAll(inTranx.keySet());
                int sSec = outTranx.size() ;

                inBr.close();outBr.close();
                int transition =0;
                int merge =0;
                int split = 0;
                for(Long txId:inTranx.keySet()){
                    //merge, split, transition
                    int icount = inTranx.get(txId).size();
                    if(outTranx.containsKey(txId)) {

                        int ocount = outTranx.get(txId).size();
                        if (icount == ocount) transition++;
                        else if (icount > ocount) merge++;
                        else split++;
                        if(icount> dim)icount= dim;
                        if(ocount> dim)ocount= dim;
                        occM[icount-1][ocount-1]++;
                        sumM[icount-1][ocount-1]+= sums.get(txId)/Math.pow(10,5);

                    }
                }
                System.out.println(year+" "+week+" "+sIn + " " + sOut+" "+sSec+" Merge:"+merge+ " Split:"+split+" Transition:"+transition);
                if(merge+split+transition>0){
                    writeMatrix(year,week,occM,crdir,"occ");
                    writeMatrix(year,week,sumM,crdir,"sum");
                }
            }
        }
    }


    private static void writeMatrix(int year, int week, int[][] occ, String dir, String file) throws Exception {
        BufferedWriter wr = new BufferedWriter(new FileWriter(dir+file+year+"week"+week+".csv"));
        StringBuffer bf = new StringBuffer();
        for(int i=0; i<occ.length; i++){
            for(int j=0; j<occ.length-1; j++){
                bf.append(occ[i][j]+",");
            }
            bf.append(occ[i][occ.length-1]);
            bf.append("\r\n");
        }
        wr.append(bf.toString());
        wr.close();
    }

    private static void loadTranx(BufferedReader inBr, int year, int week, HashMap<Long, HashSet<Long>> inTranx, HashMap<Long,Long> sums) throws IOException {
        String line;
        while((line=inBr.readLine())!=null){
            String[] arr = line.split("\t");
            Long txId = Long.parseLong(arr[0]);
            DateTime time = new DateTime(1000*Long.parseLong(arr[1]));
            int txYear = time.year().get();

            int txWeek =  time.weekOfWeekyear().get();
            if(year==txYear&&week==txWeek){
                String []adds = arr[3].split(",");
                inTranx.put(txId,new HashSet<Long>());
                for(String a:adds){
                    inTranx.get(txId).add(Long.parseLong(a));
                }
                sums.put(txId,Long.parseLong(arr[2]));
            }

        }
    }
}
