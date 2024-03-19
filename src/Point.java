public class Point {
    private final float x;
    private final float y;

    public Point(float x, float y){
        this.x = x;
        this.y = y;
    }

    public float getX(){
        return x;
    }

    public float getY(){
        return y;
    }

    public double distanceTo(Point other) {
        float deltaX = other.getX() - this.getX();
        float deltaY = other.getY() - this.getY();
        return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    }

}
