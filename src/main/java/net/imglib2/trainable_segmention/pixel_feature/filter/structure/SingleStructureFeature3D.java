
package net.imglib2.trainable_segmention.pixel_feature.filter.structure;

import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import preview.net.imglib2.algorithm.convolution.Convolution;
import preview.net.imglib2.algorithm.convolution.kernel.Kernel1D;
import preview.net.imglib2.algorithm.convolution.kernel.SeparableKernelConvolution;
import preview.net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.img.Img;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.trainable_segmention.RevampUtils;
import net.imglib2.trainable_segmention.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmention.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmention.pixel_feature.filter.hessian.EigenValuesSymmetric3D;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import org.scijava.plugin.Parameter;

import java.util.Arrays;
import java.util.List;

public class SingleStructureFeature3D extends AbstractFeatureOp {

	@Parameter
	double sigma = 1.0;

	@Parameter
	double integrationScale = 1.0;

	@Override
	public int count() {
		return 3;
	}

	@Override
	public List<String> attributeLabels() {
		return Arrays.asList("Structure_largest_" + sigma + "_" + integrationScale,
			"Structure_middle_" + sigma + "_" + integrationScale,
			"Structure_smallest_" + sigma + "_" + integrationScale);
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		final Interval targetInterval = output.get(0);
		Convolution<NumericType<?>> convolution = gaussConvolution(input.pixelSize());
		final Interval derivativeInterval = convolution.requiredSourceInterval(targetInterval);
		RandomAccessibleInterval<DoubleType> derivatives = derivatives(input, derivativeInterval);
		RandomAccessibleInterval<DoubleType> products = products(derivatives);
		RandomAccessibleInterval<DoubleType> blurredProducts = Views.interval(products,
			RevampUtils.appendDimensionToInterval(targetInterval, 0, 5));
		convolution.process(products, blurredProducts);

		EigenValuesSymmetric3D eigenvalueComputer = new EigenValuesSymmetric3D();
		LoopBuilder.setImages(Views.collapse(blurredProducts), Views.collapse(Views.stack(output)))
			.forEachPixel(eigenvalueComputer::compute);
	}

	private Convolution<NumericType<?>> gaussConvolution(double[] pixelSizes) {
		final Kernel1D[] gauss = Arrays.stream(pixelSizes)
			.mapToObj(pixelSize -> gaussKernel(integrationScale / pixelSize))
			.toArray(Kernel1D[]::new);
		return SeparableKernelConvolution.convolution(gauss);
	}

	private Kernel1D gaussKernel(double v) {
		return Kernel1D.symmetric(Gauss3.halfkernels(new double[] { v })[0]);
	}

	private RandomAccessibleInterval<DoubleType> products(
		RandomAccessibleInterval<DoubleType> input)
	{
		Interval interval = RevampUtils.removeLastDimension(input);
		RandomAccessibleInterval<DoubleType> output = ops().create().img(RevampUtils
			.appendDimensionToInterval(interval, 0, 5), new DoubleType());
		LoopBuilder.setImages(Views.collapse(input), Views.collapse(output)).forEachPixel(
			SingleStructureFeature3D::productPerPixel);
		return output;
	}

	private static void productPerPixel(Composite<DoubleType> i, Composite<DoubleType> o) {
		double x = i.get(0).getRealDouble(), y = i.get(1).getRealDouble(), z = i.get(2).getRealDouble();
		o.get(0).setReal(x * x);
		o.get(1).setReal(x * y);
		o.get(2).setReal(x * z);
		o.get(3).setReal(y * y);
		o.get(4).setReal(y * z);
		o.get(5).setReal(z * z);
	}

	private RandomAccessibleInterval<DoubleType> derivatives(FeatureInput input,
		Interval derivativeInterval)
	{
		RandomAccessibleInterval<DoubleType> gauss = ops().create().img(Intervals.expand(
			derivativeInterval, 1));
		double[] pixelSizes = input.pixelSize();
		double[] sigmas = Arrays.stream(pixelSizes).map(pixelSize -> sigma / pixelSize).toArray();
		RandomAccessible<FloatType> original = input.original();
		Gauss3.gauss(sigmas, original, gauss);
		int n = derivativeInterval.numDimensions();
		Img<DoubleType> tmp = ops().create().img(RevampUtils.appendDimensionToInterval(
			derivativeInterval, 0, n - 1), new DoubleType());
		for (int i = 0; i < n; i++)
			derive(gauss, Views.hyperSlice(tmp, n, i), i, pixelSizes[i]);
		return tmp;
	}

	private void derive(RandomAccessible<? extends RealType<?>> input,
		RandomAccessibleInterval<? extends RealType<?>> tmp, int d, double pixelSize)
	{
		final RandomAccessibleInterval<? extends RealType<?>> back = Views.interval(input, Intervals
			.translate(tmp, -1, d));
		final RandomAccessibleInterval<? extends RealType<?>> front = Views.interval(input, Intervals
			.translate(tmp, 1, d));
		final double factor = 0.5 / pixelSize;
		LoopBuilder.setImages(tmp, back, front).forEachPixel((r, b, f) -> {
			r.setReal((f.getRealDouble() - b.getRealDouble()) * factor);
		});
	}
}
