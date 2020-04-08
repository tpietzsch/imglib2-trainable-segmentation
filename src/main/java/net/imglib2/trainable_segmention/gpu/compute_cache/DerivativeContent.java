
package net.imglib2.trainable_segmention.gpu.compute_cache;

import net.imglib2.trainable_segmention.gpu.api.CLIJLoopBuilder;
import net.imglib2.trainable_segmention.gpu.api.GpuImage;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.trainable_segmention.gpu.api.GpuApi;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.trainable_segmention.gpu.api.GpuView;
import net.imglib2.trainable_segmention.gpu.api.GpuViews;
import net.imglib2.util.Intervals;

import java.util.Objects;

public class DerivativeContent implements ComputeCache.Content {

	private final ComputeCache cache;
	private final ComputeCache.Content input;
	private final int d;

	public DerivativeContent(ComputeCache cache, ComputeCache.Content input, int d) {
		this.cache = cache;
		this.input = input;
		this.d = d;
	}

	@Override
	public int hashCode() {
		return Objects.hash(input, d);
	}

	@Override
	public boolean equals(Object obj) {
		return obj instanceof DerivativeContent &&
			input.equals(((DerivativeContent) obj).input) &&
			d == ((DerivativeContent) obj).d;
	}

	@Override
	public void request(Interval interval) {
		cache.request(input, requiredInput(interval));
	}

	private FinalInterval requiredInput(Interval interval) {
		long[] border = new long[interval.numDimensions()];
		border[d] = 1;
		return Intervals.expand(interval, border);
	}

	private FinalInterval shrink(Interval interval) {
		long[] border = new long[interval.numDimensions()];
		border[d] = -1;
		return Intervals.expand(interval, border);
	}

	@Override
	public GpuImage load(Interval interval) {
		GpuApi gpu = cache.gpuApi();
		double[] pixelSize = cache.pixelSize();
		GpuView source = cache.get(input, requiredInput(interval));
		Interval center = shrink(new FinalInterval(source.dimensions()));
		GpuView front = GpuViews.crop(source, Intervals.translate(center, 1, d));
		GpuView back = GpuViews.crop(source, Intervals.translate(center, -1, d));
		GpuImage result = gpu.create(Intervals.dimensionsAsLongArray(center), NativeTypeEnum.Float);
		CLIJLoopBuilder.gpu(gpu)
			.addInput("f", front)
			.addInput("b", back)
			.addInput("factor", 0.5 / pixelSize[d])
			.addOutput("r", result)
			.forEachPixel("r = (f - b) * factor");
		return result;
	}
}