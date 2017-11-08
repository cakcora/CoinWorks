import edu.uci.ics.jung.graph.DirectedSparseGraph;
import edu.uci.ics.jung.graph.util.Pair;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by cxa123230 on 10/24/2017.
 * <p>
 * requires data from @BackwardBranchExtractor.java
 */
public class GraphCreator {
    public static void main(String[] args) throws IOException {
        for (int maxWave = 2; maxWave <= 3; maxWave++)
            for (String dataType : new String[]{"blackmail", "white"})
                for (String graphType : new String[]{"address", "transaction"}) {
                    System.out.println(maxWave + " " + graphType + " " + dataType);
                    computeGraph(dataType, graphType, maxWave);
                }
    }

    private static void computeGraph(String typeofData, String graphType, int maxWave) throws IOException {
        String d2 = "D:/Bitcoin/createddata/subchain/";
        int i = 0;
        String fileName = d2 + typeofData + maxWave + "wave" + graphType + "graph.txt";
        BufferedWriter wr = new BufferedWriter(new FileWriter(fileName));

        DirectedSparseGraph<String, Long> graph = new DirectedSparseGraph<>();
        BufferedReader fr = new BufferedReader(new FileReader(d2 + typeofData + "transactions.txt"));
        if (graphType.equals("address")) {
            findGraph(2, 3, maxWave, graph, fr);
        }
        if (graphType.equals("transaction")) {
            findGraph(3, 2, maxWave, graph, fr);
        }
        writeGraph(wr, graph);
        wr.close();
        fr.close();
    }

    private static void findGraph(int edgeIndex, int nodeIndex, int maxWave, DirectedSparseGraph<String, Long> graph, BufferedReader fr) throws IOException {
        String line;
        HashMap<String, HashSet<String>> inHolder = new HashMap<>();
        HashMap<String, HashSet<String>> outHolder = new HashMap<>();
        while ((line = fr.readLine()) != null) {
            String arr[] = line.split("\t");
            String type = arr[0];
            int wave = Integer.parseInt(arr[1]);
            if (wave > maxWave) continue;
            String edge = arr[edgeIndex];
            String node = arr[nodeIndex];
            if (type.equals("i")) {
                if (!inHolder.containsKey(edge)) inHolder.put(edge, new HashSet<>());
                inHolder.get(edge).add(node);
                graph.addVertex(node);
            } else if (type.equals("o")) {
                if (!outHolder.containsKey(edge)) outHolder.put(edge, new HashSet<>());
                outHolder.get(edge).add(node);
                graph.addVertex(node);
            }
        }
        System.out.println("vertices " + graph.getVertexCount());
        for (String edgeVal : inHolder.keySet()) {
            if (outHolder.containsKey(edgeVal)) {
                for (String node1 : inHolder.get(edgeVal)) {
                    for (String node2 : outHolder.get(edgeVal)) {
//                        if(node1.equals(node2)) System.out.println(node1+" the same in "+edgeVal);
                        graph.addEdge((long) graph.getEdgeCount(), node1, node2);
                    }
                }
            }
        }
    }

    private static void writeGraph(BufferedWriter wr, DirectedSparseGraph<String, Long> graph) throws IOException {


        HashMap<String, Integer> id = new HashMap<>();

        for (Long e : graph.getEdges()) {
            Pair<String> f = graph.getEndpoints(e);
            long n1 = getId(id, f.getFirst());
            long n2 = getId(id, f.getSecond());

            wr.write(n1 + "\t" + n2 + "\r\n");
        }
        wr.close();

        HashSet<String> addresses = new HashSet<String>();
        addresses.add("13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94");
        addresses.add("12t9YDPgwueZ9NyMgw519p7AA8isjr6SMw");
        addresses.add("115p7UMMngoj1pMvkpHijcRdfJNXj6LrLn");

        for (String ad : addresses) {
            System.out.println(ad + "\tblackaddress_graph_node_id\t" + id.get(ad));
        }
        addresses.clear();
        addresses.add("1GUkazUBpXWdSJ9HbgTapAH7uybpi3Cs6K");
        addresses.add("14wXrm49HxggbdQ6RGfWY8qghGEWhLA28K");
        addresses.add("1BSw11nrTuxwt8YyX3Mv6sPsAXRM29MYdc");

        for (String ad : addresses) {
            System.out.println(ad + "\twhiteaddress_graph_node_id\t" + id.get(ad));
        }
        System.out.println("edges " + graph.getEdgeCount());
    }

    private static int getId(HashMap<String, Integer> id, String node) {
        if (!id.containsKey(node)) id.put(node, id.size());
        return id.get(node);
    }


}
