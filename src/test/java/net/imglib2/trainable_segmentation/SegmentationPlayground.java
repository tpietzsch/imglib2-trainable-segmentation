
package net.imglib2.trainable_segmentation;

import bdv.util.BdvFunctions;
import hr.irb.fastRandomForest.FastRandomForest;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.trainable_segmentation.classification.Segmenter;
import net.imglib2.trainable_segmentation.classification.Trainer;
import net.imglib2.trainable_segmentation.gpu.random_forest.RFAnalysis;
import net.imglib2.trainable_segmentation.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.trainable_segmentation.utils.SingletonContext;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;
import preview.net.imglib2.loops.LoopBuilder;

public class SegmentationPlayground
{
	public static void main( String... args )
	{
		System.out.println("starting...");
		final String fnImage = "drosophila_3d.tif";
		final String fnLabeling = "drosophila_3d_labeling.tif";

		final RandomAccessibleInterval< FloatType > image = Utils.loadImageFloatType( fnImage );
		final LabelRegions< ? > labelRegions = asLabelRegions( Utils.loadImageFloatType( fnLabeling ) );

		final Context context = SingletonContext.getInstance();

		final FeatureSettings featureSettings = new FeatureSettings( GlobalSettings.default3d().build(),
				GroupedFeatures.gauss(),
				GroupedFeatures.differenceOfGaussians(),
				GroupedFeatures.hessian(),
				GroupedFeatures.gradient() );

		final Segmenter segmenter = Trainer.train( context, image, labelRegions, featureSettings );

		///

		final ArrayImg< UnsignedByteType, ? > segmentation = ArrayImgs.unsignedBytes( Intervals.dimensionsAsLongArray( image ) );

		RandomAccessibleInterval<FloatType> featureValues = segmenter.features().apply( Views.extendBorder( image ), segmentation );
		final RFAnalysis.RFPrediction prediction = RFAnalysis.analyze(
				( FastRandomForest ) segmenter.getClassifier(),
				segmenter.features().count() );
		prediction.segment( featureValues, segmentation );
		BdvFunctions.show( segmentation, "segmentation" );

		///

		System.out.println( "done" );
	}

	// -- Helper --

	private static LabelRegions asLabelRegions( RandomAccessibleInterval< ? extends RealType< ? > > labeling )
	{
		Img< UnsignedByteType > ints = ArrayImgs.unsignedBytes( Intervals.dimensionsAsLongArray( labeling ) );
		RandomAccessibleInterval< LabelingType< String > > labelingTypes = new ImgLabeling<>( ints );
		LoopBuilder.setImages( labeling, labelingTypes ).multiThreaded().forEachPixel( ( i, o ) -> {
			if ( i.getRealFloat() != 0 )
				o.add( i.toString() );
		} );
		return new LabelRegions<>( labelingTypes );
	}
}
