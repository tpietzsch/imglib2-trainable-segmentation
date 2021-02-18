package net.imglib2.trainable_segmentation.gpu.random_forest;

import hr.irb.fastRandomForest.FastRandomForest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

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

}
