package Anomaly;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import edu.uci.ics.jung.graph.DirectedSparseGraph;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Collection;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.HashSet;

public class Windowed {

    private static final Logger logger = LoggerFactory.getLogger(Windowed.class);
    static int txLength = 64;
    static int window = 1;
    static double bitcoin = Math.pow(10, 8);
    static double lower = 0.2;

    public static void main(String args[]) throws IOException {
        BufferedWriter wr = new BufferedWriter(new FileWriter("graphComps.csv"));


        String dataDir = args[0];
        String suspAddFile = args[1];
        DirectedSparseGraph<Object, WeightedEdge> graph = new DirectedSparseGraph();
        HashMap<Output, Integer> outputs = new HashMap<>();
        HashSet<String> suspAdd = read(suspAddFile);
        BiMap<String, Integer> nodeIDs = HashBiMap.create();
        long passed = 0;
        for (int year = 2016; year < 2018; year++) {
            for (int day = 49; day <= 365; day++) {
                graph = new DirectedSparseGraph();
                nodeIDs.clear();
                String fPath = dataDir + "outputs" + year + "Day" + day + ".txt";
                if (!new File(fPath).exists()) continue;
                String oPath = dataDir + "inputs" + year + "Day" + day + ".txt";
                addDayOutputs(graph, outputs, nodeIDs, fPath);
                addDayInputs(graph, outputs, nodeIDs, oPath);
                int k = 0;
                passed++;
                if (passed == window) {
                    logger.info(">" + graph.getVertexCount() + " vertices, " + graph.getEdgeCount() + " edges");
                    filterEdges(graph);
                    logger.info(graph.getVertexCount() + " vertices, " + graph.getEdgeCount() + " edges");

                    process2(wr, graph, suspAdd, nodeIDs, year, day);
                    passed = 0;
                    graph = new DirectedSparseGraph();
                    nodeIDs.clear();
                    outputs.clear();
                }
            }

        }
        wr.close();
    }

    private static void filterEdges(DirectedSparseGraph<Object, WeightedEdge> graph) {
        for (WeightedEdge edge : new HashSet<>(graph.getEdges())) {
            if (edge.getAmount() < lower * bitcoin) {
                graph.removeEdge(edge);
            }
        }
        for (Object node : new HashSet<>(graph.getVertices())) {
            if (graph.getNeighborCount(node) == 0) {
                graph.removeVertex(node);
            }
        }
    }


    private static void process2(BufferedWriter wr, DirectedSparseGraph<Object, WeightedEdge> graph, HashSet<String> suspAdd, BiMap<String, Integer> nodeIDs, int year, int day) throws IOException {

        HashMap<Integer, Double> weights = new HashMap<>();
        HashMap<Integer, Integer> lengths = new HashMap<>();
        int count = 0;
        int tx = 0, add = 0;
        for (Object n1 : graph.getVertices()) {
//            logger.info(weights.size()+" visited vertices");
            if (n1.toString().length() != txLength) {
                add++;
                continue;
            }
            tx++;
            if (graph.getInEdges(n1).size() != 0) continue;
            HashSet<Integer> seen = new HashSet(4);
            boolean err = traverse(0, 1.0, lengths, weights, graph, n1, seen);
            if (err) { //this should not happen. Recursion goes wrong.
                logger.error("Recursion gone wrong, returning");
            }
        }
        BufferedWriter w = new BufferedWriter(new FileWriter(year + "_" + day + "dayWeights.txt"));
        w.append("node\tlength\tweight\tneighbors\tincome\tlabel\r\n");
        for (Object n : weights.keySet()) {
            int nn = Integer.parseInt(n.toString());

            if (weights.get(nn) > 1 || suspAdd.contains(n)) {
                int label = 0;
                // logger.info(n+" has \t"+weights.get(n));
                String addr = nodeIDs.inverse().get(n);
                if (suspAdd.contains(addr)) {
                    logger.info("suspicious");
                    label = 1;
                }
                long amo = 0;
                for (WeightedEdge we : graph.getInEdges(nn)) {
                    amo += we.getAmount();
                }
                w.append(nodeIDs.inverse().get(nn) + "\t" + lengths.get(nn) + "\t" + weights.get(nn) + "\t" + graph.getNeighborCount(nn) + "\t" + amo + "\t" + label + "\r\n");
                count++;
            }
        }
        w.close();


        DescriptiveStatistics ds = new DescriptiveStatistics();
        for (Object o : graph.getVertices()) {
            String s = o.toString();
            if (s.length() != txLength) {
                ds.addValue(graph.getNeighborCount(o));
            }
        }
        String msg = year + "\t" + day + "\t" + count + "\t" + graph.getVertexCount() + "\t" + graph.getEdgeCount() + "\t" + ds.getMean() + "\t" + ds.getMax() + "\t" + ds.getN();
        logger.info(msg);
        wr.append(msg + "\r\n");
    }


    private static HashSet<String> read(String suspAddFile) throws IOException {
        String line = "";
        HashSet<String> ad = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(suspAddFile));
        br.readLine();//header
        while ((line = br.readLine()) != null) {
            String add = line.split("\t")[4];
            ad.add(add);
        }
        return ad;
    }

    private static boolean traverse(int len, double weight, HashMap<Integer, Integer> lengths, HashMap<Integer, Double> weights, DirectedSparseGraph graph, Object n1, HashSet<Integer> seen) {
        if (len > (window * 24 * 6)) {
            return true;
        }
        Collection outEdges = new HashSet(graph.getSuccessors(n1));
        HashSet<Object> remove = new HashSet<>();
        if (outEdges == null) return false;
        int s = outEdges.size();
        if ((len % 2) == 0) {
            //update the map
            for (Object N2 : outEdges) {
                int n2 = Integer.parseInt(N2.toString());
                if (seen.contains(n2)) {
                    remove.add(N2);
                }
                if (!weights.containsKey(n2)) {
                    weights.put(n2, 0.0);
                    lengths.put(n2, 0);
                }
                weights.put(n2, weights.get(n2) + weight / s);
                int length = lengths.get(n2) + 1;
                if (len > length)
                    lengths.put(n2, len);
                seen.add(n2);
            }
        }

        outEdges.removeAll(remove);
        //seen.addAll(outEdges);
//        if(outEdges.size()>15)
        //logger.info("function call times:"+outEdges.size());
        for (Object N2 : outEdges) {
            traverse(len + 1, weight / s, lengths, weights, graph, N2, seen);
        }
        return false;
    }

    private static void addDayInputs(DirectedSparseGraph graph, HashMap<Output, Integer> outputs, BiMap<String, Integer> nodeIDds, String oPath) throws IOException {
        String line;
        BufferedReader br = new BufferedReader(new FileReader(oPath));
        int nodeCount = graph.getVertexCount();
        int edgeNumber = graph.getEdgeCount();
        while ((line = br.readLine()) != null) {
            String arr[] = line.split("\t");
            String tx = arr[1];
            for (int j = 2; j < arr.length; j = j + 2) {
                String prevTx = arr[j];

                if (tx.length() != txLength || prevTx.length() != txLength) {
                    logger.error("Tx id appears to be less than " + txLength + " chars in this line: " + line);
                    continue;
                }

                int index = Integer.parseInt(arr[j + 1]);
                Output key = new Output(prevTx, index);
                long amo;
                int nID;
                if (outputs.containsKey(key)) {
                    nID = outputs.get(key);
                    amo = ((WeightedEdge) graph.findEdge(prevTx, nID)).getAmount();
                    WeightedEdge edge1 = new WeightedEdge(amo, edgeNumber);
                    if (graph.addEdge(edge1, nID, tx)) {
                        edgeNumber++;
                    }
                } else {
                    //address("unknown");
                    amo = -1;
                    graph.addVertex(prevTx);
                    nID = nodeCount;
                    graph.addVertex(nID);
                    nodeIDds.put(key.toString(), nID);
                    nodeCount++;
                    WeightedEdge edge1 = new WeightedEdge(amo, edgeNumber);
                    if (graph.addEdge(edge1, prevTx, nID)) {
                        edgeNumber++;
                    }
                    WeightedEdge edge2 = new WeightedEdge(amo, edgeNumber);
                    if (graph.addEdge(edge2, nID, tx)) {
                        edgeNumber++;
                    }
                }
            }
        }
    }

    private static void addDayOutputs(DirectedSparseGraph graph, HashMap<Output, Integer> outputs, BiMap<String, Integer> nodeIDds, String fPath) throws IOException {

        String line;
        BufferedReader br = new BufferedReader(new FileReader(fPath));
        int nodeCount = graph.getVertexCount();
        int edgeNumber = graph.getEdgeCount();
        while ((line = br.readLine()) != null) {
            String arr[] = line.split("\t");
            if (arr.length % 2 != 0) continue;
            String tx = arr[1];
            if (tx.length() != txLength) {
                logger.error("Wrong tx id" + tx);
                System.exit(1);
            }
            //address to the graph


            graph.addVertex(tx);
            for (int j = 2; j < arr.length; j = j + 2) {
                String address = arr[j];
                if (!address.equalsIgnoreCase("noaddress")) {
                    int index = (j - 2) / 2;
                    long amo = Long.parseLong(arr[j + 1]);
                    Output out = new Output(tx, address, index, amo);
                    if (!nodeIDds.containsKey(address)) {
                        graph.addVertex(nodeCount);
                        nodeIDds.put(address, nodeCount);
                        nodeCount++;
                    }
                    int addID = nodeIDds.get(address);
                    WeightedEdge edge = new WeightedEdge(amo, edgeNumber);
                    if (graph.addEdge(edge, tx, addID)) {
                        edgeNumber++;
                    }
                    outputs.put(out, addID);

                }
            }

        }
    }

    private static long inSec(int year, int month, int day) {
        return new GregorianCalendar(year, month, day).getTime().getTime() / 1000;
    }

    private static int getId(HashMap<String, Integer> idMap, String tx, int id) {
        idMap.put(tx, id);
        return idMap.get(tx);
    }
}
