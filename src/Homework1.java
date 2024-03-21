import java.io.*;
import java.util.*;

public class Homework1{

    public static void main(String[] args) throws IOException {

        // Check if the number of arguments is correct
        System.out.println(args.length);
        if(args.length != 5)
            throw new IllegalArgumentException("Wrong number of params!");

        // Read all points in the file and add them to the  (integers), and must compute and print the following information.list of points
        Scanner scanner =    new Scanner(new File(args[0]));
        List<Point> points = new ArrayList<>();
        while (scanner.hasNextLine()) {
            String[] cords = scanner.nextLine().split(",");
            points.add(new Point(Float.parseFloat(cords[0]), Float.parseFloat(cords[1])));
        }

        // Read D, M, K parameters
        float D = Float.parseFloat(args[1]);
        int M = 1; // Default
        int K = 1; // Default

        try{
            M = Integer.parseInt(args[2]);
        }catch(NumberFormatException e){
            throw new IllegalArgumentException("Param M is not an integer!");
        }

        try{
            K = Integer.parseInt(args[3]);
        }catch(NumberFormatException e){
            throw new IllegalArgumentException("Param K is not an integer!");
        }

        //Add the control for param L in the task2

        ExactOutliers(points, D, M, K);

    }

    public static void ExactOutliers(List<Point> points, float D, int M, int K) {
        ArrayList<Point> outliers = new ArrayList<>();

        // Calculate the distances
        // for a total of N(N-1)/2 distances being calculated
        double[][] distances = new double[points.size()][points.size()];
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                distances[i][j] = points.get(i).distanceTo(points.get(j));
                distances[j][i] = points.get(i).distanceTo(points.get(j));
            }
        }


        // Counting close (<= D) points for each point and if they are < M then
        // the point gets saved into a list as outlier
        for (int i = 0; i < points.size(); i++) {
            int count = 0;
            for (int j = 0; j < points.size(); j++)
                if (distances[i][j] <= D)
                    count++;

            if (count < M) {
                outliers.add(points.get(i));
            }
        }

        // Sort outliers based on number of closest points
        outliers.sort((p1, p2) -> {
            int count1 = 0, count2 = 0;
            int index1 = points.indexOf(p1);
            int index2 = points.indexOf(p2);
            for (int j = 0; j < points.size(); j++) {
                if (distances[index1][j] <= D)
                    count1++;

                if (distances[index2][j] <= D)
                    count2++;

            }
            return Integer.compare(count1, count2);
        });

        // Print the number of (D,M)-outliers
        System.out.println("Number of (D, M)-outliers: " + outliers.size());

        // Print the first K outliers
        System.out.println("First " + K + " outliers:");
        for (int i = 0; i < Math.min(K, outliers.size()); i++)
            System.out.println("(" + outliers.get(i).getX() + ", " + outliers.get(i).getY() + ")");
    }

}