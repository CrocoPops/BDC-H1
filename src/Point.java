import org.sparkproject.dmg.pmml.False;

public class Point {
    private final float x;
    private final float y;

    public Point(float x, float y){
        this.x = x;
        this.y = y;
    }

    public float getX(){
        return this.x;
    }

    public float getY(){
        return this.y;
    }

    public double distanceTo(Point other) {
        float deltaX = other.getX() - this.getX();
        float deltaY = other.getY() - this.getY();
        // return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        return Math.sqrt(Math.pow(deltaX, 2) + Math.pow(deltaY, 2));
    }

}
