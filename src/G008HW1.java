import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.jetbrains.annotations.NotNull;
import scala.Tuple2;
import scala.Tuple3;
import java.io.*;
import java.util.*;


public class G008HW1 {

    public static void main(String[] args) throws IOException {
        // Check if the number of arguments is correct
        if (args.length != 5)
            throw new IllegalArgumentException("Wrong number of params!");

        // Variables
        float D = Float.parseFloat(args[1]);
        int M = Integer.parseInt(args[2]);
        int K = Integer.parseInt(args[3]);
        int L = Integer.parseInt(args[4]);

        // Time measurement
        long startTime;
        long endTime;
        long totalTime;

        // Printing CLI arguments
        System.out.println(args[0] + " D=" + args[1] + " M=" + args[2] + " K=" + args[3] + " L=" + args[4] + " ");

        // Read all points in the file and add them to the list
        Scanner scanner = new Scanner(new File(args[0]));
        List<Point> points = new ArrayList<>();
        while (scanner.hasNextLine()) {
            String[] cords = scanner.nextLine().split(",");
            points.add(new Point(Float.parseFloat(cords[0]), Float.parseFloat(cords[1])));
        }

        // Print the number of points
        System.out.println("Number of points = " + points.size());

        // Check if number of points is lower than 200000 and if so
        // compute ExactOutliers
        if(points.size() <= 200000) {
            startTime = System.currentTimeMillis();
            ExactOutliers(points, D, M, K);
            endTime = System.currentTimeMillis();
            totalTime = endTime - startTime;
            System.out.println("Running time of ExactOutliers = " + totalTime + "ms");
        }

        // Creating the Spark context and calling outliers approximate computation
        SparkConf conf = new SparkConf(true).setAppName("OutlierDetector");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            sc.setLogLevel("ERROR");
            // Divide the inputFile in L partitions (each line is assigned to a specific partition
            JavaRDD<String> docs = sc.textFile(args[0]).repartition(L).cache();
            startTime = System.currentTimeMillis();
            MRApproxOutliers(docs, D, M, K);
            endTime   = System.currentTimeMillis();
            totalTime = endTime - startTime;
            System.out.println("Running time of MRApproxOutliers  = " + totalTime + "ms");
        }
    }
    public static void MRApproxOutliers(JavaRDD<String> docs, float D, int M, int K) {
        // ROUND 1
        // Mapping each pair (X,Y) into ((X,Y), 1)
        JavaPairRDD<Tuple2<Integer, Integer>, Long> cell = docs.flatMapToPair(document -> { // <-- MAP PHASE (R1)
            String[] cord = document.split(",");

            // Compute the cells coordinates
            double lambda = D/(2*Math.sqrt(2));

            // Finding cell coordinates
            Tuple2<Integer, Integer> cellCoordinates = new Tuple2<>(
                    (int) Math.floor(Float.parseFloat(cord[0]) / lambda),
                    (int) Math.floor(Float.parseFloat(cord[1]) / lambda)
            );

            return Collections.singletonList(new Tuple2<>(cellCoordinates, 1L)).iterator();
        }).reduceByKey(Long::sum).cache();

        // Saving the non-empty cells in a local structure
        Map<Tuple2<Integer, Integer>, Long> nonEmptyCells = cell.collectAsMap();

        // Round 2
        // Adding information on N3 and N7 for each K-V pair
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> cellNeighbors = cell.mapToPair(pair -> {
            int i = pair._1()._1();
            int j = pair._1()._2();
            long totalCount = pair._2();
            long N3 = 0L;
            long N7 = 0L;

            for(int dx = -3; dx <= 3; dx++) {
                for(int dy = -3; dy <= 3; dy++) {
                    Tuple2<Integer, Integer> neighborKey = new Tuple2<>(i + dx, j + dy);
                    Long neighborCount = nonEmptyCells.get(neighborKey);

                    if(neighborCount != null){
                        if ((Math.abs(dx) <= 1) && (Math.abs(dy) <= 1))
                            N3 += neighborCount;
                        N7 += neighborCount;
                    }
                }
            }

            Tuple3<Long, Long, Long> counts = new Tuple3<>(totalCount, N3, N7);
            return new Tuple2<>(new Tuple2<>(i, j), counts);
        }).cache();

        // Number of sure (D, M) - outliers
        long sureOutliers = 0;
        for(Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._3() <= M).collect())
            sureOutliers += i._2()._1();
        System.out.println("Number of sure outliers = " + sureOutliers);

        // Number of uncertain points
        long uncertainOutliers = 0;
        for(Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._2() <= M && triple._2()._3() > M).collect())
            uncertainOutliers += i._2()._1();
        System.out.println("Number of uncertain points = " + uncertainOutliers);

        // First K non-empty cells in non-decreasing order of N3
        List<Tuple2<Long, Tuple2<Tuple2<Integer, Integer>, Long>>> topKCells = cell.mapToPair(
                tuple -> new Tuple2<>(tuple._2(), tuple)
        ).sortByKey(true).take(K);

        for (Tuple2<Long, Tuple2<Tuple2<Integer, Integer>, Long>> i_cell : topKCells)
            System.out.println("Cell: " + i_cell._2()._1() + " Size = " + i_cell._1());

    }

    public static void ExactOutliers(List<Point> points, float D, int M, int K) {
        // Create array to contain points and their respective close points (<= D)
        PointsNearby[] counts = new PointsNearby[points.size()];

        // Initialize the data-structure for each point
        // Initializing at 1 because we have to consider the point itself
        for (int i = 0; i < points.size(); i++)
            counts[i] = new PointsNearby(points.get(i), 1L);

        // Compute how many points are close (<= D) for each point
        // Using the symmetry we make N(N-1)/2 calculations instead of N^2
        for(int i = 0; i < points.size(); i++)
            for(int j = i + 1; j < points.size(); j++)

                // If we find a point which distance is lower than D, we update both
                // the points used in the calculation (because of symmetry)
                if(points.get(i).distanceTo(points.get(j)) <= Math.pow(D, 2)){
                    counts[i].nearby += 1;
                    counts[j].nearby += 1;
                }

        // Find the outliers if nearby points (closer than D) are less than M
        List<Point> outliers = new ArrayList<>();
        for (int i = 0; i < points.size(); i++)
            if(counts[i].nearby <= M)
                outliers.add(points.get(i));

        // Print the number of (D,M)-outliers
        System.out.println("Number of Outliers = " + outliers.size());

        // Print the first K outliers
        Arrays.sort(counts);
        for (int i = 0; i < Math.min(K, outliers.size()); i++)
            System.out.printf("Point: (%.4f, %.4f)\n", outliers.get(i).x, outliers.get(i).y);
    }
}

// Class used as struct to contain information about the points
class Point {
    double x;
    double y;

    public Point(double x, double y){
        this.x = x;
        this.y = y;
    }

    public double distanceTo(Point other) {
        double deltaX = other.x - this.x;
        double deltaY = other.y - this.y;
        return Math.pow(deltaX, 2) + Math.pow(deltaY, 2);
    }

}

// Class used as struct to contain information about a point and its neighbours
// Implements the comparable interface, so we can sort the array (counts)
class PointsNearby implements Comparable<PointsNearby>{
    public Point p;
    public Long nearby;

    public PointsNearby(Point p, Long nearby){
        this.p = p;
        this.nearby = nearby;
    }

    @Override
    public int compareTo(@NotNull PointsNearby o) {
        return Long.compare(this.nearby, o.nearby);
    }
}


