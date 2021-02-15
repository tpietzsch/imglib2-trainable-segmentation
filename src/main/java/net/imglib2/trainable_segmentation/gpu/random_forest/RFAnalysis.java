package net.imglib2.trainable_segmentation.gpu.random_forest;

import Jama.Matrix;
import hr.irb.fastRandomForest.FastRandomForest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.BenchmarkHelper;
import net.imglib2.util.StopWatch;
import net.imglib2.view.composite.Composite;
import preview.net.imglib2.loops.LoopBuilder;

public class RFAnalysis
{
	public static RFPrediction analyze( final FastRandomForest classifier, final int numberOfFeatures )
	{
		final TransparentRandomForest forest = new TransparentRandomForest( classifier );



		final Map< Integer, List< TransparentRandomTree > > heightHist = new HashMap<>();
		for ( TransparentRandomTree tree : forest.trees() )
			heightHist.computeIfAbsent( tree.height(), ArrayList::new ).add( tree );
		for ( Integer i : heightHist.keySet() )
			System.out.println( "trees with height " + i + ": " + heightHist.get( i ).size() );
		System.out.println();



		final Map< Integer, List< TransparentRandomTree > > numNodesHist = new HashMap<>();
		for ( TransparentRandomTree tree : forest.trees() )
			numNodesHist.computeIfAbsent( tree.numberOfNodes(), ArrayList::new ).add( tree );
		for ( Integer i : numNodesHist.keySet() )
			System.out.println( "trees with " + i + " nodes: " + numNodesHist.get( i ).size() );
		System.out.println();

		final AtomicInteger numLeafs = new AtomicInteger( 0 );
		forest.forEachNode( t -> {
			if ( t.isLeaf() )
				numLeafs.incrementAndGet();
		} );
		System.out.println( "numLeafs.intValue() = " + numLeafs.intValue() );


//		int sumThresholds = 0;
//		for ( int i = 0; i < numberOfFeatures; i++ )
//		{
//			final int featureIndex = i;
//			final Set< Double > thresholds = new HashSet<>();
////			final List< Double > thresholds = new ArrayList<>();
//			forest.forEachNode( t -> {
//				if ( t.attributeIndex() == featureIndex )
//					thresholds.add( t.threshold() );
//			} );
//			System.out.println( "feature " + i + ": " + thresholds.size() + " thresholds" );
//			sumThresholds += thresholds.size();
//		}
//		System.out.println( "sumThresholds = " + sumThresholds );
//		System.out.println();



		final Set< List< Double > > probs = new HashSet<>();
		forest.forEachNode( t -> {
			if ( t.isLeaf() )
			{
				List< Double > list = new ArrayList<>();
				for ( double d : t.classProbabilities() )
					list.add( d );
				probs.add( list );
			}
		} );
		System.out.println( "probs.size() = " + probs.size() );
		System.out.println();

		return new RFPrediction( forest, numberOfFeatures );
	}


	public static class RFPrediction
	{
		// every entry is 2 floats: feature index, threshold
		private final float[] dataTrees;

		// every entry is numClasses floats
		private final float[] probabilities;

		private final int numTrees;

		private final int numFeatures;

		private final int numClasses;

		// TODO at index i contains numbers of trees of/up to height i + 1
		private final int[] numTreesUpToHeight;

		public RFPrediction( FastRandomForest classifier, int numberOfFeatures )
		{
			this( new TransparentRandomForest( classifier ), numberOfFeatures );
		}

		public RFPrediction( final TransparentRandomForest forest, int numberOfFeatures )
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
			final int maxLeafs = 1 << maxHeight;
			final int maxNonLeafs = maxLeafs - 1;

			dataTrees = new float[ 2 * maxNonLeafs * numTrees ];
			probabilities = new float[ numClasses * maxLeafs * numTrees ];

			// TODO at index i contains numbers of trees of/up to height i + 1
			numTreesUpToHeight = new int[ 1 ];
			for ( int i = 0; i < numTreesUpToHeight.length; ++i )
				numTreesUpToHeight[ i ] = treesByHeight.getOrDefault( i + 1, Collections.emptyList() ).size();

			int iTreeBase = 0;
			for ( int height : heights )
			{
				final List< TransparentRandomTree > trees = treesByHeight.get( height );
				for ( TransparentRandomTree tree : trees )
					write( tree, iTreeBase++, 0, 0, 0, maxHeight - 1 );
			}
		}

		/**
		 * Applies the random forest to each pixel in the feature stack. Write the index
		 * of the class with the highest probability into the output image.
		 *
		 * @param featureStack Input image. Axis order should be XYZC of XYC. Number of
		 *          channels must equal {@link #numFeatures}.
		 * @param out Output image. Axis order should be XYZ or XY. Pixel values will be
		 *          between 0 and {@link #numClasses} - 1.
		 */
		public void segment( RandomAccessibleInterval< FloatType > featureStack,
				RandomAccessibleInterval<? extends IntegerType<?> > out)
		{
			BenchmarkHelper.benchmarkAndPrint( 20, true, () -> {
//				StopWatch watch = StopWatch.createAndStart();
				LoopBuilder.setImages( FastViews.collapse( featureStack ), out ).forEachChunk( chunk -> {
					float[] features = new float[ numFeatures ];
					float[] probabilities = new float[ numClasses ];
					chunk.forEachPixel( ( featureVector, classIndex ) -> {
						copyFromTo( featureVector, features );
						distributionForInstance( features, probabilities );
						classIndex.setInteger( ArrayUtils.findMax( probabilities ) );
					} );
					return null;
				} );
//				System.out.println( "(t) segment runtime " + watch );
			} );
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
			final int numClasses = this.numClasses;

			Arrays.fill( distribution, 0 );
			final int nh1 = numTreesUpToHeight[ 0 ];
			final int nhmax = this.numTrees;
			final int maxLeafs = 2 << maxDepth;
			final int maxNonLeafs = maxLeafs - 1;
			final int dataSize = maxNonLeafs * 2;
			final int probSize = maxLeafs * numClasses;
			int dataBase = 0;
			int probBase = 0;
			for ( int tree = 0; tree < nh1; ++tree )
			{
				addDistributionForTree_h1( instance, distribution, dataBase, probBase, maxDepth, numClasses );
				dataBase += dataSize;
				probBase += probSize;
			}
			for ( int tree = nh1; tree < nhmax; ++tree )
			{
				addDistributionForTree( instance, distribution, dataBase, probBase, maxDepth, numClasses );
				dataBase += dataSize;
				probBase += probSize;
			}
//			for ( int tree = 0; tree < numTrees; ++tree )
//				addDistributionForTree( instance, tree, distribution, maxDepth );
			ArrayUtils.normalize( distribution );
		}

		private void addDistributionForTree(
				final float[] instance,
				final float[] distribution,
				final int dataBase,
				final int probBase,
				final int maxDepth,
				final int numClasses )
		{
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
					branchBits = ( branchBits << 1 ) + branch;
				}
			}

			final int o = probBase + branchBits * numClasses;
			for ( int k = 0; k < numClasses; k++ )
				distribution[ k ] += probabilities[ o + k ];
		}

		private void addDistributionForTree_h1(
				final float[] instance,
				final float[] distribution,
				final int dataBase,
				final int probBase,
				final int maxDepth,
				final int numClasses )
		{
			final int attributeIndex = ( int ) dataTrees[ dataBase ];
			final float attributeValue = instance[ attributeIndex ];
			final float threshold = dataTrees[ dataBase + 1 ];
			if( attributeValue < threshold )
			{
				final int p = probBase + ( attributeValue < threshold ? 0 : numClasses << maxDepth );
				for ( int k = 0; k < numClasses; k++ )
					distribution[ k ] += probabilities[ p + k ];
			}
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
				if ( depth < maxDepth )
				{
					// mark as leaf by setting feature index to -1 or something
					final int o = dataBase + nodeIndex * 2;
					dataTrees[ o ] = -1;
				}
				final int o = probBase + ( branchBits << ( maxDepth - depth ) ) * numClasses;
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
}
