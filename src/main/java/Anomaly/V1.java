package Anomaly;

import edu.uci.ics.jung.algorithms.cluster.WeakComponentClusterer;
import edu.uci.ics.jung.algorithms.filters.FilterUtils;
import edu.uci.ics.jung.graph.DirectedSparseGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.HashSet;
import java.util.concurrent.ThreadLocalRandom;

public class V1 {
    private static final Logger logger = LoggerFactory.getLogger(V1.class);

    public static void main(String[] args) {
        int e = 0;
        logger.info("time enters");


        int nodec = 1000;
        for (int f = 0; f < 10; f++) {
            DirectedSparseGraph graph = new DirectedSparseGraph();
            for (int i = 0; i < nodec; i++) {
                for (int j = 0; j < 500; j++) {
                    int randomNum = ThreadLocalRandom.current().nextInt(0, nodec);
                    graph.addEdge(e++, i, j);
                }
            }
            WeakComponentClusterer wcc = new WeakComponentClusterer();
            Collection<DirectedSparseGraph> ccs = FilterUtils.createAllInducedSubgraphs(wcc.transform(graph), graph);
        }
        logger.info("time exist 1");
        for (int f = 0; f < 10; f++) {
            DirectedSparseGraph graph = new DirectedSparseGraph();
            for (int i = 0; i < nodec / 2; i++) {
                for (int j = 0; j < 250; j++) {
                    int randomNum = ThreadLocalRandom.current().nextInt(0, nodec / 2);
                    graph.addEdge(e++, i, j);
                }
            }

            for (int i = nodec / 2 + 1; i < nodec; i++) {
                for (int j = 0; j < 250; j++) {
                    int randomNum = ThreadLocalRandom.current().nextInt(nodec / 2 + 1, nodec);
                    graph.addEdge(e++, i, j);
                }
            }
            while (graph.getVertexCount() > 0) {
                DirectedSparseGraph newGraph = new DirectedSparseGraph();
                logger.info("enters");
                Object g = new HashSet(graph.getVertices()).iterator().next();

                HashSet<Object> next = new HashSet<>();
                next.addAll(graph.getNeighbors(g));
                while (!next.isEmpty()) {
                    Object o = next.iterator().next();
                    next.addAll(graph.getNeighbors(o));
                    graph.removeVertex(o);
                    next.remove(o);
                    newGraph.addEdge(0, g, o);
                    // System.out.println("removing " + o);
                }

            }
        }
        logger.info("time exist 2");
    }
}
