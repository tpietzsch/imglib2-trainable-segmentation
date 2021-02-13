
package net.imglib2.trainable_segmentation.gpu.random_forest;

import java.util.function.Consumer;
import weka.core.Instance;

/**
 * Wraps around a hr.irb.fastRandomForest.FastRandomTree and exposes all the
 * parameters (threshold, children and classification probabilities).
 * <p>
 * Java reflection is used to extract the parameter from the weka fast random
 * forest.
 */
public class TransparentRandomTree
{

	private final int attribute;

	private final double threshold;

	private final TransparentRandomTree smallerChild;

	private final TransparentRandomTree biggerChild;

	private final double[] classProbabilities;

	/**
	 * @param fastRandomTree
	 * 		is expected to be of type
	 * 		hr.irb.fastRandomForest.FastRandomTree
	 */
	public TransparentRandomTree( Object fastRandomTree )
	{
		this.attribute = ReflectionUtils.getPrivateField( fastRandomTree, "m_Attribute", Integer.class );
		if ( isLeaf() )
		{
			this.threshold = Double.NaN;
			this.smallerChild = null;
			this.biggerChild = null;
			this.classProbabilities = ReflectionUtils.getPrivateField( fastRandomTree, "m_ClassProbs",
					double[].class );
		}
		else
		{
			this.threshold = ReflectionUtils.getPrivateField( fastRandomTree, "m_SplitPoint",
					Double.class );
			Object[] sucessors = ReflectionUtils.getPrivateField( fastRandomTree, "m_Successors",
					Object[].class );
			this.smallerChild = new TransparentRandomTree( sucessors[ 0 ] );
			this.biggerChild = new TransparentRandomTree( sucessors[ 1 ] );
			this.classProbabilities = null;
		}
	}

	public int attributeIndex()
	{
		return attribute;
	}

	public double threshold()
	{
		return threshold;
	}

	/**
	 * Returns the sub tree that is used if the attribute value is smaller than the
	 * threshold.
	 */
	public TransparentRandomTree smallerChild()
	{
		return smallerChild;
	}

	/**
	 * Returns the sub tree that is used if the attribute value is greater or equal
	 * to the threshold.
	 */
	public TransparentRandomTree biggerChild()
	{
		return biggerChild;
	}

	public double[] classProbabilities()
	{
		return classProbabilities;
	}

	public double[] distributionForInstance( Instance instance )
	{
		if ( !isLeaf() )
		{
			TransparentRandomTree child = instance.value( attribute ) < threshold ? smallerChild
					: biggerChild;
			return child.distributionForInstance( instance );
		}
		else
		{
			return classProbabilities;
		}
	}

	/**
	 * Returns true if the tree is only one leafnode.
	 */
	public boolean isLeaf()
	{
		return attribute == -1;
	}

	public int height()
	{
		return isLeaf() ? 0 : 1 + Math.max( smallerChild().height(), biggerChild().height() );
	}

	public int numberOfNodes()
	{
		return 1 + ( isLeaf() ? 0 : smallerChild().numberOfNodes() + biggerChild().numberOfNodes() );
	}

	public void forEachNode( Consumer< TransparentRandomTree > visitor )
	{
		visitor.accept( this );
		if ( !isLeaf() )
		{
			smallerChild.forEachNode( visitor );
			biggerChild.forEachNode( visitor );
		}
	}

	public int numberOfClasses() {
		if (isLeaf())
			return classProbabilities.length;
		else
			return smallerChild.numberOfClasses();
	}
}
