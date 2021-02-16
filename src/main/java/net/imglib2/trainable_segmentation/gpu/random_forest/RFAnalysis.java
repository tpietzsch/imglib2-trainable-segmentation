package net.imglib2.trainable_segmentation.gpu.random_forest;

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
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.real.FloatType;
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
		System.out.println( "numLeafs = " + numLeafs.intValue() );


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
		// every entry is 2 values: feature index, threshold
		private final short[] attributes;
		private final float[] thresholds;

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

			final int maxHeight = heights.length == 0 ? 0 : heights[ heights.length - 1 ];
			numTreesUpToHeight = new int[ maxHeight ];
			int totalDataSize = 0;
			int totalProbSize = 0;
			for ( int i = 0; i < numTreesUpToHeight.length; ++i )
			{
				numTreesUpToHeight[ i ] = treesByHeight.getOrDefault( i + 1, Collections.emptyList() ).size();
				final int numLeafs = 2 << i;
				final int numNonLeafs = numLeafs - 1;
				totalDataSize += numNonLeafs * numTreesUpToHeight[ i ];
				totalProbSize += numLeafs * numClasses * numTreesUpToHeight[ i ];
			}

			attributes = new short[ totalDataSize ];
			thresholds = new float[ totalDataSize ];
			probabilities = new float[ totalProbSize ];

			int dataBase = 0;
			int probBase = 0;
			for ( int height : heights )
			{
				final int depth = height - 1;
				final int numLeafs = 2 << depth;
				final int numNonLeafs = numLeafs - 1;
				final int dataSize = numNonLeafs;
				final int probSize = numLeafs * numClasses;
				final List< TransparentRandomTree > trees = treesByHeight.get( height );
				for ( TransparentRandomTree tree : trees )
				{
					write( tree, 0, 0, 0, height - 1, dataBase, probBase );
					dataBase += dataSize;
					probBase += probSize;
				}
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
//			BenchmarkHelper.benchmarkAndPrint( 20, false, () -> {
				StopWatch watch = StopWatch.createAndStart();
//				AtomicInteger ii = new AtomicInteger();
				LoopBuilder.setImages( FastViews.collapse( featureStack ), out ).forEachChunk( chunk -> {
					float[] features = new float[ numFeatures ];
					float[] probabilities = new float[ numClasses ];
					chunk.forEachPixel( ( featureVector, classIndex ) -> {
						copyFromTo( featureVector, features );
						distributionForInstance( features, probabilities );
//						distributionForInstance_DUMMY( features, probabilities );
//						final int i = ii.getAndIncrement();
//						if ( i < 3 )
//						{
//							System.out.println( "i = " + i + ": " + Arrays.toString( probabilities ) + " : " + Arrays.toString( features ) );
//						}
						classIndex.setInteger( ArrayUtils.findMax( probabilities ) );
					} );
					return null;
				} );
				System.out.println( "(t) segment runtime " + watch );
//			} );
		}

		private static void copyFromTo( final Composite< FloatType > input, final float[] output )
		{
			for ( int i = 0, len = output.length; i < len; i++ )
				output[ i ] = input.get( i ).getRealFloat();
		}

		public void distributionForInstance_DUMMY(
				float[] instance,
				float[] distribution )
		{
			Arrays.fill( distribution, 0 );
			for ( int i = 0; i < numFeatures; i+=2 )
				distribution[ 0 ] += instance[ i ];
			for ( int i = 1; i < numFeatures; i+=2 )
				distribution[ 1 ] += instance[ i ];
			ArrayUtils.normalize( distribution );
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
			switch ( numClasses )
			{
			case 2:
				distributionForInstance_c2( instance, distribution );
				break;
			case 3:
				distributionForInstance_c3( instance, distribution );
				break;
			default:
				distributionForInstance_ck( instance, distribution );
				break;
			}
		}

		public void distributionForInstance_ck(
				float[] instance,
				float[] distribution )
		{
			Arrays.fill( distribution, 0 );
			final int numClasses = this.numClasses;
			int dataBase = 0;
			int probBase = 0;
			for ( int depth = 0; depth < numTreesUpToHeight.length; depth++ )
			{
				final int nh = numTreesUpToHeight[ depth ];
				if ( nh == 0 )
					continue;

				if ( depth == 0 ) // special case for trees of height 1
				{
					final int dataSize = 1;
					final int probSize = 2 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h1( instance, dataBase );
						acc( distribution, numClasses, probBase, branchBits );
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else if ( depth == 1 ) // special case for trees of height 2
				{
					final int dataSize = 3;
					final int probSize = 4 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h2( instance, dataBase );
						acc( distribution, numClasses, probBase, branchBits );
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else // general case
				{
					final int numLeafs = 2 << depth;
					final int numNonLeafs = numLeafs - 1;
					final int dataSize = numNonLeafs;
					final int probSize = numLeafs * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree( instance, dataBase, depth );
						acc( distribution, numClasses, probBase, branchBits );
						dataBase += dataSize;
						probBase += probSize;
					}
				}
			}
			ArrayUtils.normalize( distribution );
		}

		private void acc( final float[] distribution, final int numClasses, final int probBase, final int branchBits )
		{
			for ( int k = 0; k < numClasses; k++ )
				distribution[ k ] += probabilities[ probBase + branchBits * numClasses + k ];
		}

		public void distributionForInstance_c2(
				float[] instance,
				float[] distribution )
		{
			float c0 = 0, c1 = 0;
			final int numClasses = 2;
			int dataBase = 0;
			int probBase = 0;
			for ( int depth = 0; depth < numTreesUpToHeight.length; depth++ )
			{
				final int nh = numTreesUpToHeight[ depth ];
				if ( nh == 0 )
					continue;

				if ( depth == 0 ) // special case for trees of height 1
				{
					final int dataSize = 1;
					final int probSize = 2 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h1( instance, dataBase );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else if ( depth == 1 ) // special case for trees of height 2
				{
					final int dataSize = 3;
					final int probSize = 4 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h2( instance, dataBase );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else // general case
				{
					final int numLeafs = 2 << depth;
					final int numNonLeafs = numLeafs - 1;
					final int dataSize = numNonLeafs;
					final int probSize = numLeafs * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree( instance, dataBase, depth );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
			}
			final float invsum = 1f / ( c0 + c1 );
			distribution[ 0 ] = c0 * invsum;
			distribution[ 1 ] = c1 * invsum;
		}

		public void distributionForInstance_c3(
				float[] instance,
				float[] distribution )
		{
			float c0 = 0, c1 = 0, c2 = 0;
			final int numClasses = 3;
			int dataBase = 0;
			int probBase = 0;
			for ( int depth = 0; depth < numTreesUpToHeight.length; depth++ )
			{
				final int nh = numTreesUpToHeight[ depth ];
				if ( nh == 0 )
					continue;

				if ( depth == 0 ) // special case for trees of height 1
				{
					final int dataSize = 1;
					final int probSize = 2 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h1( instance, dataBase );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						c2 += probabilities[ probBase + branchBits * numClasses + 2 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else if ( depth == 1 ) // special case for trees of height 2
				{
					final int dataSize = 3;
					final int probSize = 4 * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree_h2( instance, dataBase );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						c2 += probabilities[ probBase + branchBits * numClasses + 2 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
				else // general case
				{
					final int numLeafs = 2 << depth;
					final int numNonLeafs = numLeafs - 1;
					final int dataSize = numNonLeafs;
					final int probSize = numLeafs * numClasses;
					for ( int tree = 0; tree < nh; ++tree )
					{
						final int branchBits = addDistributionForTree( instance, dataBase, depth );
						c0 += probabilities[ probBase + branchBits * numClasses ];
						c1 += probabilities[ probBase + branchBits * numClasses + 1 ];
						c2 += probabilities[ probBase + branchBits * numClasses + 2 ];
						dataBase += dataSize;
						probBase += probSize;
					}
				}
			}
			final float norm = 1f / ( c0 + c1 + c2 );
			distribution[ 0 ] = c0 * norm;
			distribution[ 1 ] = c1 * norm;
			distribution[ 2 ] = c2 * norm;
		}

		private int addDistributionForTree(
				final float[] instance,
				final int dataBase,
				final int maxDepth )
		{
			int branchBits = 0;
			for ( int nodeIndex = 0, depth = 0; depth <= maxDepth; ++depth )
			{
				final int o = dataBase + nodeIndex;
				final int attributeIndex = attributes[ o ];
				if ( attributeIndex < 0 )
				{
					branchBits = branchBits << ( 1 + maxDepth - depth );
					break;
				}
				else
				{
					final float attributeValue = instance[ attributeIndex ];
					final float threshold = thresholds[ o ];
					final int branch = attributeValue < threshold ? 0 : 1;
					nodeIndex = ( nodeIndex << 1 ) + branch + 1;
					branchBits = ( branchBits << 1 ) + branch;
				}
			}
			return branchBits;
		}

		private int addDistributionForTree_h1(
				final float[] instance,
				final int dataBase )
		{
			final int attributeIndex = attributes[ dataBase ];
			final float attributeValue = instance[ attributeIndex ];
			final float threshold = thresholds[ dataBase ];
			final int branchBits = attributeValue < threshold ? 0 : 1;
			return branchBits;
		}

		private int addDistributionForTree_h2(
				final float[] instance,
				final int dataBase )
		{
			final int attributeIndex0 = attributes[ dataBase ];
			final float attributeValue0 = instance[ attributeIndex0 ];
			final float threshold0 = thresholds[ dataBase ];

			int branchBits = ( attributeValue0 < threshold0 ) ? 0 : 2;
			final int dataBase1 = dataBase + ( ( attributeValue0 < threshold0 ) ? 1 : 2 );
			final int attributeIndex1 = attributes[ dataBase1 ];
			if ( attributeIndex1 >= 0 )
			{
				final float attributeValue1 = instance[ attributeIndex1 ];
				final float threshold1 = thresholds[ dataBase1 ];
				if ( attributeValue1 >= threshold1 )
					branchBits += 1;
			}
			return branchBits;
		}

		private void write( TransparentRandomTree node, final int nodeIndex, final int branchBits, final int depth, final int maxDepth, final int treeDataBase, final int treeProbBase )
		{
			if ( node.isLeaf() )
			{
				// TODO
				//   write probabilities
				//   if ( depth < maxDepth ) mark as leaf by setting feature index to -1 or something
				//   if ( depth < maxDepth ) fill branchBits with zeros
				//   index in probabilities = branchBits * numClasses + base_offset?
				//   base_offset = numClasses * maxLeafs * treeIndex
//					node.classProbabilities(); // double[]
				final int b;
				if ( depth <= maxDepth )
				{
					// mark as leaf by setting feature index to -1 or something
					final int o = treeDataBase + nodeIndex;
					attributes[ o ] = -1;
					b = branchBits << ( 1 + maxDepth - depth );
				}
				else
					b = branchBits;
				final int o = treeProbBase + b * numClasses;
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
				final int o = treeDataBase + nodeIndex;
				attributes[ o ] = ( short ) node.attributeIndex();
				thresholds[ o ] = ( float ) node.threshold();

				// recurse to children:
				write( node.smallerChild(),
						( nodeIndex << 1 ) + 1,
						( branchBits << 1 ),
						depth + 1, maxDepth, treeDataBase, treeProbBase );
				write( node.biggerChild(),
						( nodeIndex << 1 ) + 2,
						( branchBits << 1 ) + 1,
						depth + 1, maxDepth, treeDataBase, treeProbBase );
			}
		}
	}
}
