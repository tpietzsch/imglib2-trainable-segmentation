package net.imglib2.trainable_segmentation.gpu.random_forest;

import hr.irb.fastRandomForest.FastRandomForest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.StopWatch;
import net.imglib2.view.composite.Composite;

public class RFPrediction_1
{
	// every entry is 2 floats: feature index, threshold
	private final float[] dataTrees;

	// every entry is numClasses floats
	private final float[] probabilities;

	private final int numTrees;

	private final int numFeatures;

	private final int numClasses;

	public RFPrediction_1( FastRandomForest classifier, int numberOfFeatures )
	{
		this( new TransparentRandomForest( classifier ), numberOfFeatures );
	}

	public RFPrediction_1( final TransparentRandomForest forest, int numberOfFeatures )
	{
		numClasses = forest.numberOfClasses();
		numTrees = forest.trees().size();
		numFeatures = numberOfFeatures;

		final Map< Integer, List< TransparentRandomTree > > treesByHeight = new HashMap<>();
		for ( TransparentRandomTree tree : forest.trees() )
			treesByHeight.computeIfAbsent( tree.height(), ArrayList::new ).add( tree );
		final int[] heights = treesByHeight.keySet().stream().mapToInt( Integer::intValue ).sorted().toArray();
		for ( int i : heights )
			System.out.println( "trees with height " + i + ": " + treesByHeight.get( i ).size() );
		System.out.println();

		final int maxHeight = heights.length == 0 ? 0 : heights[ heights.length - 1 ];
//		final int maxHeight = forest.trees().stream().mapToInt( TransparentRandomTree::height ).max().orElse( 0 );
		final int maxLeafs = 1 << maxHeight;
		final int maxNonLeafs = maxLeafs - 1;

		dataTrees = new float[ 2 * maxNonLeafs * numTrees ];
		probabilities = new float[ numClasses * maxLeafs * numTrees ];

		int iTree = 0;
		for ( int height : heights )
		{
			final List< TransparentRandomTree > trees = treesByHeight.get( height );
			for ( TransparentRandomTree tree : trees )
				write( tree, iTree++, 0, 0, 0, maxHeight - 1 );
		}

//		for ( int iTree = 0; iTree < numTrees; ++iTree )
//			write( forest.trees().get( iTree ), iTree, 0, 0, 0, maxHeight - 1 );
	}

	/**
	 * Applies the random forest to each pixel in the feature stack. Write the index
	 * of the class with the highest probability into the output image.
	 *
	 * @param featureStack
	 * 		Input image. Axis order should be XYZC of XYC. Number of
	 * 		channels must equal {@link #numFeatures}.
	 * @param out
	 * 		Output image. Axis order should be XYZ or XY. Pixel values will be
	 * 		between 0 and {@link #numClasses} - 1.
	 */
	public void segment( RandomAccessibleInterval< FloatType > featureStack,
			RandomAccessibleInterval< ? extends IntegerType< ? > > out )
	{
		StopWatch watch = StopWatch.createAndStart();
		AtomicInteger ii = new AtomicInteger();
		LoopBuilder.setImages( FastViews.collapse( featureStack ), out ).forEachChunk( chunk -> {
			float[] features = new float[ numFeatures ];
			float[] probabilities = new float[ numClasses ];
			chunk.forEachPixel( ( featureVector, classIndex ) -> {
				copyFromTo( featureVector, features );
				distributionForInstance( features, probabilities );
				final int i = ii.getAndIncrement();
				if ( i < 10 )
				{
					System.out.println( "i = " + i + ": " + Arrays.toString( probabilities ) + " : " + Arrays.toString( features ) );
				}
				classIndex.setInteger( ArrayUtils.findMax( probabilities ) );
			} );
			return null;
		} );
		System.out.println( "(t) segment runtime " + watch );
	}

	private static void copyFromTo( final Composite< FloatType > input, final float[] output )
	{
		for ( int i = 0, len = output.length; i < len; i++ )
			output[ i ] = input.get( i ).getRealFloat();
	}

	/**
	 * Applies the random forest to the given instance. Writes the class
	 * probabilities to the parameter called distribution.
	 *
	 * @param instance
	 * 		Instance / feature vector, must be an array of length
	 *        {@code numberOfFeatures}.
	 * @param distribution
	 * 		This is the output buffer, array length mush equal
	 *        {@code numClasses}.
	 */
	public void distributionForInstance(
			float[] instance,
			float[] distribution )
	{
		final int maxDepth = 3; // TODO

		Arrays.fill( distribution, 0 );
//			for ( int tree = 0, n = numTrees; tree < n; tree++ )
		for ( int tree = 0; tree < numTrees; ++tree )
			addDistributionForTree( instance, tree, distribution, maxDepth );
		ArrayUtils.normalize( distribution );
	}

	private void addDistributionForTree(
			final float[] instance,
			final int treeIndex,
			final float[] distribution,
			final int maxDepth )
	{
		final int maxLeafs = 2 << maxDepth;
		final int maxNonLeafs = maxLeafs - 1;
		final int dataBase = treeIndex * maxNonLeafs * 2;
		final int probBase = treeIndex * maxLeafs * numClasses;

		int branchBits = 0;
		for ( int nodeIndex = 0, depth = 0; depth <= maxDepth; ++depth )
		{
			final int o = dataBase + nodeIndex * 2;
			final int attributeIndex = ( int ) dataTrees[ o ];
			if ( attributeIndex < 0 )
			{
				branchBits = branchBits << ( maxDepth - depth );
				break;
			}
			else
			{
				final float attributeValue = instance[ attributeIndex ];
				final float threshold = dataTrees[ o + 1 ];
				final int branch = attributeValue < threshold ? 0 : 1;
				nodeIndex = ( nodeIndex << 1 ) + branch + 1;
				branchBits = ( branchBits << 1 ) + branch;
			}
		}

		final int o = probBase + branchBits * numClasses;
		for ( int k = 0; k < numClasses; k++ )
			distribution[ k ] += probabilities[ o + k ];
	}

	private void write( TransparentRandomTree node, final int treeIndex, final int nodeIndex, final int branchBits, final int depth, final int maxDepth )
	{
		final int maxLeafs = 2 << maxDepth;
		final int maxNonLeafs = maxLeafs - 1;
		final int dataBase = treeIndex * maxNonLeafs * 2;
		final int probBase = treeIndex * maxLeafs * numClasses;

		if ( node.isLeaf() )
		{
			// TODO
			//   write probabilities
			//   if ( depth < maxDepth ) mark as leaf by setting feature index to -1 or something
			//   if ( depth < maxDepth ) fill branchBits with zeros
			//   index in probabilities = branchBits * numClasses + base_offset?
			//   base_offset = numClasses * maxLeafs * treeIndex
//				node.classProbabilities(); // double[]
			final int b;
			if ( depth <= maxDepth )
			{
				// mark as leaf by setting feature index to -1 or something
				final int o = dataBase + nodeIndex * 2;
				dataTrees[ o ] = -1;
				b = branchBits << ( maxDepth - depth );
			}
			else
				b = branchBits;
			final int o = probBase + b * numClasses;
			for ( int i = 0; i < numClasses; ++i )
				probabilities[ o + i ] = ( float ) node.classProbabilities()[ i ];
		}
		else // not a leaf
		{
			// TODO
			//   write feature index and threshold
			//     at ( treeIndex * maxNonLeafs + nodeIndex ) * 2
			//   recurse to children:
			//     left:
			//       smallerChild(), treeIndex, (nodeIndex << 1) + 1, (branchBits << 1), depth + 1, maxDepth
			//     right:
			//       biggerChild(), treeIndex, (nodeIndex << 1) + 2, (branchBits << 1) + 1, depth + 1, maxDepth
//				node.attributeIndex() // int
//				node.threshold() // double
//				node.smallerChild() TransparentRandomTree

			// write feature index and threshold
			final int o = dataBase + nodeIndex * 2;
			dataTrees[ o ] = node.attributeIndex();
			dataTrees[ o + 1 ] = ( float ) node.threshold();

			// recurse to children:
			write( node.smallerChild(), treeIndex,
					( nodeIndex << 1 ) + 1,
					( branchBits << 1 ),
					depth + 1, maxDepth );
			write( node.biggerChild(), treeIndex,
					( nodeIndex << 1 ) + 2,
					( branchBits << 1 ) + 1,
					depth + 1, maxDepth );
		}
	}
}
