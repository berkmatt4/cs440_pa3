package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.pas.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.tetris.nn.layers.Tanh;
import edu.bu.pas.tetris.nn.layers.Sigmoid;
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int numPixelsInImage = 4;
        final int hiddenDim = 4 * numPixelsInImage;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix flattenedImage = null;
        Matrix rowVector = null;
        try
        {
            flattenedImage = game.getGrayscaleImage(potentialAction);

            //store num rows and columns in the game board
            int numCols = flattenedImage.getShape().getNumCols();
            int numRows = flattenedImage.getShape().getNumRows();

            /* we can start by checking how full the game board itself is */
            double [] howFull = new double[numCols];        //storing how full each column is
            for (int i = 0; i < numCols; i ++){
                double numMinos = 0;
                for (int j = 0; j < numRows; j++){

                    // if value at point in image is more than 0, there is a block there
                    if(flattenedImage.get(j, i) > 0){
                        numMinos += 1;
                    }
                }
                howFull[i] = (double)numMinos;
            }

            /* following from fullness, we can check the total space left in a given column */
            double [] spaceLeft = new double[numCols];
            for (int i = 0; i < numCols; i ++){
                spaceLeft[i] = numRows - howFull[i];        //space left in col should be (num rows - rows filled) 
            }

            /* finding holes
             * a hole would be where an empty space is surrounded entirely
             * by other tetris blocks
             */

             double [] holes = new double[numCols];
             for( int i = 0; i < numCols; i ++){
                holes[i] = 0.0;        //init the number of holes to 0

                /*start at the top of the highest block in each column
                 * then, work downward, noting anywhere that the value of the gameboard
                 * is < 0.5 indicating there is a hole
                 */
                for( int j = (int)(spaceLeft[i] - 1); j >= 0; j --){
                    if(flattenedImage.get(numRows - j - 1, i) < 0.5){
                        //if the space is empty, but there is a block above it, we have a hole
                        holes[i] += 1;
                    }
                }
             }

             /* another measurement: the bumpiness
              * this represents the differences in heights for adjascent columns
              * this is important because columns that are taller should be priority
              * to break down so we don't lose
              */

             double [] bumps = new double[numCols];
             for(int i = 0; i < numCols; i ++){
                bumps[i] = 0.0;       //start filling with 0s
             }
             for( int i = 1; i < numCols; i ++){
                double hDiff = Math.abs(spaceLeft[i] - spaceLeft[i-1]);     //differences in heights
                bumps[i] = (double)hDiff;
             }

             //now we need to get the maximums of those values to put in the row vector
             double maxHeight = Double.MIN_VALUE;
             double totalHoles = 0.0;
             double maxBumps = Double.MIN_VALUE;
             double rowsCleared = 0.0;

             //collecting the values
             for(double height : spaceLeft){maxHeight = Math.max(maxHeight, height);}
             for(double bump : bumps){maxBumps = Math.max(maxBumps, bump);}
             for(double hole : holes){totalHoles += hole;}
             
             for(int i = 0; i < numRows; i ++){
                boolean willBeCleared = true;
                
                //check if each space in the row will be filled
                for(int j = 0; j < numCols; j++){
                    willBeCleared = willBeCleared & (flattenedImage.get(i, j) > 0); //take the & of the two statements to confirm whole row is full
                }
                if(willBeCleared){
                    rowsCleared += 1;
                }
             }

             rowVector = Matrix.zeros(1, 4); //init final row vector with the 4 params we have created
             rowVector.set(0, 0, maxHeight);
             rowVector.set(0, 1, totalHoles);
             rowVector.set(0, 2, rowsCleared);
             rowVector.set(0, 3, maxBumps);

        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }

        return rowVector;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // System.out.println("cycleIdx=" + gameCounter.getCurrentCycleIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        /* we will decay exploration rate over time
         * this will be done by factoring in the current cycle and current game
         */
        double cycleProgress = Math.min(1.0, gameCounter.getCurrentCycleIdx() / 1000.0);
        double gameProgress = Math.min(1.0, gameCounter.getCurrentGameIdx() / 10000.0);
        double progressFactor = (cycleProgress + gameProgress) / 2.0;       //always between 1 and 0

        //decay from 0.9 to 0.05 over time
        double explroationProb = Math.max(0.05, 0.9 - (0.85 * progressFactor));

        return this.getRandom().nextDouble() <= explroationProb;        //randomize but less than the exploration prob
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
    
        if (possibleMoves.isEmpty()) {
            // Fallback if no moves available
            Pair<Mino, Double> optimalMove = getBestActionAndQValue(game);
            return optimalMove.getFirst();
        }
        
        // Heuristic-based exploration (80% of the time)
        if (this.getRandom().nextDouble() < 0.8) {
            // Evaluate each move with a simple heuristic and pick the best one
            Mino bestMove = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (Mino move : possibleMoves) {
                try {
                    // Get a temporary view of what the board would look like
                    Matrix tempView = game.getGrayscaleImage(move);
                    int numRows = tempView.getShape().getNumRows();
                    int numCols = tempView.getShape().getNumCols();
                    
                    // Count potential line clears
                    int potentialLines = 0;
                    for (int row = 0; row < numRows; row++) {
                        boolean rowFull = true;
                        for (int col = 0; col < numCols; col++) {
                            if (tempView.get(row, col) <= 0) {
                                rowFull = false;
                                break;
                            }
                        }
                        if (rowFull) {
                            potentialLines++;
                        }
                    }
                    
                    // Simple score based on lines cleared
                    double moveScore = Math.pow(10, potentialLines);
                    
                    // Penalize height
                    double maxHeight = 0;
                    for (int col = 0; col < numCols; col++) {
                        int height = 0;
                        for (int row = 0; row < numRows; row++) {
                            if (tempView.get(row, col) > 0) {
                                height = numRows - row;
                                break;
                            }
                        }
                        maxHeight = Math.max(maxHeight, height);
                    }
                    moveScore -= maxHeight * 2;
                    
                    // Choose the move with the best score
                    if (moveScore > bestScore) {
                        bestScore = moveScore;
                        bestMove = move;
                    }
                } catch (Exception e) {
                    // Skip if we can't evaluate this move
                    continue;
                }
            }
            
            // If we found a good move, use it
            if (bestMove != null) {
                return bestMove;
            }
        }
        
        // Otherwise, just pick a random move (20% of the time or if heuristic failed)
        int randomIndex = this.getRandom().nextInt(possibleMoves.size());
        return possibleMoves.get(randomIndex);
    }


    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a cycle, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each cycle.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        // Base reward starts at a small negative value to encourage action
        double reward = -1.0;
        
        // Add score-based reward
        reward += game.getScoreThisTurn();
        
        try {
            Board board = game.getBoard();
            int numRows = Board.NUM_ROWS;
            int numCols = Board.NUM_COLS;
            
            // COLUMN HEIGHT CALCULATION
            double[] columnHeights = new double[numCols];
            for (int col = 0; col < numCols; col++) {
                int height = 0;
                for (int row = 0; row < numRows; row++) {
                    if (board.isCoordinateOccupied(col, row)) {
                        height = numRows - row;
                        break;
                    }
                }
                columnHeights[col] = height;
            }
            
            // HOLES CALCULATION - Much more aggressive detection
            double totalHoles = 0;
            for (int col = 0; col < numCols; col++) {
                int highestBlock = -1;
                
                // Find the highest block in this column
                for (int row = 0; row < numRows; row++) {
                    if (board.isCoordinateOccupied(col, row)) {
                        highestBlock = row;
                        break;
                    }
                }
                
                // Count holes below the highest block
                if (highestBlock >= 0) {
                    for (int row = highestBlock + 1; row < numRows; row++) {
                        if (!board.isCoordinateOccupied(col, row)) {
                            totalHoles++;
                        }
                    }
                }
            }
            
            // LINES CLEARED - Direct calculation
            int linesCleared = 0;
            for (int row = 0; row < numRows; row++) {
                boolean rowFull = true;
                for (int col = 0; col < numCols; col++) {
                    if (!board.isCoordinateOccupied(col, row)) {
                        rowFull = false;
                        break;
                    }
                }
                if (rowFull) {
                    linesCleared++;
                }
            }
            
            // BUMPINESS - Difference between adjacent columns
            double bumpiness = 0;
            for (int col = 0; col < numCols - 1; col++) {
                bumpiness += Math.abs(columnHeights[col] - columnHeights[col + 1]);
            }
            
            // WELL CREATION - Reward for creating a "well" for long pieces
            double wellReward = 0;
            for (int col = 0; col < numCols; col++) {
                // Check if this column is at least 3 blocks lower than both neighbors
                boolean isWell = true;
                
                if (col > 0) {
                    isWell = isWell && (columnHeights[col] + 3 <= columnHeights[col-1]);
                }
                if (col < numCols - 1) {
                    isWell = isWell && (columnHeights[col] + 3 <= columnHeights[col+1]);
                }
                
                if (isWell) {
                    wellReward += 20.0; // Substantial reward for creating a well
                }
            }
            
            // Calculate aggregate height
            double totalHeight = 0;
            double maxHeight = 0;
            for (double height : columnHeights) {
                totalHeight += height;
                maxHeight = Math.max(maxHeight, height);
            }
            
            // EXTREME REWARD for line clears
            reward += Math.pow(10, linesCleared) * 20;
            
            // BIG PENALTY for holes
            reward -= totalHoles * 20.0;
            
            // Moderate penalty for bumpiness
            reward -= bumpiness * 2.0;
            
            // Penalty for high stack
            reward -= maxHeight * 1.5;
            
            // Penalty for aggregate height
            reward -= totalHeight * 0.5;
            
            // Add well reward
            reward += wellReward;
            
            // Extreme penalty for game over
            if (game.didAgentLose()) {
                reward -= 10000.0;
            }
            
        } catch (Exception e) {
            System.out.println("Exception in getReward: " + e.getMessage());
        }
        
        return reward;
    }
}
