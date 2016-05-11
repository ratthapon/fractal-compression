package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;

import th.ac.kmitl.it.prip.fractal.Parameters;

public class Compressor {
	private static final Logger LOGGER = Logger.getLogger(Compressor.class
			.getName());

	private Parameters parameters = null;
	private float[] data = null;
	private final int[] parts;
	private int nParts;
	private int nSamples;
	private AtomicInteger partProgress;
	private AtomicInteger samplesProgress;
	private long initTime;
	private long completeTime;
	private boolean isDone = false;

	public Compressor(float[] inputAudioData, Parameters compressParameters) {
		parameters = compressParameters;
		data = inputAudioData;
		nSamples = data.length;
		parts = this.partition();
		nSamples = Arrays.stream(parts).sum();
		nParts = parts.length;
		samplesProgress = new AtomicInteger(0);
		partProgress = new AtomicInteger(0);
	}

	private boolean needToSegment(int from, int to) {
		// equivalent http://www.mathworks.com/help/matlab/ref/var.html

		// check if no. of elements under segment bound
		if (to >= nSamples) {
			return true;
		}
		// check if chunk size over max block size
		if (to - from + 1 > parameters.getMaxBlockSize()) {
			return true;
		}
		// check if chunk size under min block size
		if (to - from + 1 < parameters.getMinBlockSize()
				* parameters.getDomainScale()) {
			return false;
		}
		// find mean
		float sumOfChunk = 0.0f;
		for (int i = from; i < to; i++) {
			sumOfChunk = sumOfChunk + data[i] / 32768.0f;
		}
		float mu = sumOfChunk / (to - from + 1);
		// find variance
		sumOfChunk = 0.0f;
		for (int i = from; i < to; i++) {
			sumOfChunk = sumOfChunk
					+ (float) Math.pow((data[i] / 32768.0f) - mu, 2);
		}
		float var = sumOfChunk / (to - from);

		// decide to segment
		boolean result = var > parameters.getThresh();
		return result;
	}

	private int[] partition() {
		int[] rangeSize = new int[nSamples]; // partitioned boundary size
		boolean canPartition = true;

		rangeSize[0] = nSamples; // init size
		// init size
		for (int idx = 0; idx < nSamples; idx += parameters.getMaxBlockSize()) {
			rangeSize[idx] = parameters.getMaxBlockSize();

			// more segment to cover all data
			if (nSamples <= idx + rangeSize[idx]) {
				int chunkSize = parameters.getMaxBlockSize();
				int chunkIdx = idx;
				while (chunkSize >= parameters.getMinBlockSize()) {
					if (chunkIdx + chunkSize >= nSamples) {
						chunkSize = chunkSize / 2;
					} else {
						rangeSize[chunkIdx] = chunkSize;
						chunkIdx = chunkIdx + chunkSize;
					}
				}
			}
		}

		// loop until can't partition
		canPartition = true;
		while (canPartition) {
			canPartition = false;

			// for entire data
			for (int idx = 0; idx < nSamples; idx++) {
				int chunkSize = 0;
				chunkSize = (int) rangeSize[idx];
				// found boundary information
				if (chunkSize > 0) {
					// retrieve necessary information
					int from = idx;
					int to = from + chunkSize - 1;
					// check if need to partition
					if (needToSegment(from, to)) {
						canPartition = true;

						// reset old information in boundary
						for (int i = from; i < to && i < nSamples; i++) {
							rangeSize[i] = 0;
						}
						// set new information
						int midPosition = from
								+ (int) Math.floor((to - from) / 2) + 1;
						if (midPosition < nSamples) {
							rangeSize[idx] = midPosition - from;
						}
						if (to < nSamples && midPosition < nSamples) {
							rangeSize[midPosition] = to - midPosition + 1;
						}
					}
				}
			}
		}
		// count non zero for allocate buffer
		int count = 0;
		for (int i = 0; i < rangeSize.length; i++) {
			int chunkSize = (int) rangeSize[i];
			if (chunkSize > 0) {
				count = count + 1;
			}
		}
		// remove zero value
		int[] result = new int[count];
		int reduceIdx = 0;
		for (int i = 0; i < rangeSize.length; i++) {
			int chunkSize = (int) rangeSize[i];
			if (chunkSize > 0) {
				result[reduceIdx] = (int) (rangeSize[i]);
				reduceIdx = reduceIdx + 1;
			}
		}
		return result;
	}

	public double[][] compress() {
		final int nCoeff = parameters.getNCoeff();
		double[][] code = new double[nParts][nCoeff + 3];
		ExecutorService executorService = Executors
				.newWorkStealingPool(parameters.getMaxParallelProcess());
		List<Callable<float[]>> rangeTask = new ArrayList<Callable<float[]>>();

		// each range block
		int rbIdx = 0;
		for (int fIdx = 0; fIdx < nParts; fIdx++) {
			final float rangeIdx = fIdx;
			// locate range block
			final int bColStart = rbIdx;
			final int rangeBlockSize = (int) (parts[fIdx]);
			final int bColEnd = rbIdx + rangeBlockSize - 1;
			rbIdx = rbIdx + (int) parts[fIdx]; // cumulative for next range

			// parallel range mapping : queuing phase
			rangeTask.add(new Callable<float[]>() {

				public float[] call() throws Exception {
					float[] codeChunk = new float[nCoeff + 3];
					float bestR = Float.POSITIVE_INFINITY;
					// block y
					float[] b = Arrays
							.copyOfRange(data, bColStart, bColEnd + 1);

					// search similarity matched domain form entire data
					for (int dbIdx = 0; dbIdx < nSamples - rangeBlockSize
							* parameters.getDomainScale() - 1; dbIdx += parameters
							.getDStep()) {

						// locate test domain
						int dColStart = dbIdx;
						int dColEnd = dbIdx + rangeBlockSize
								* parameters.getDomainScale() - 1;
						// input x
						float[] d = Arrays.copyOfRange(data, dColStart,
								dColEnd + 1);

						// reduce to half size
						float[] a = resample(d);
						for (int rev = 0; rev <= 1; rev++) {
							if (rev == 1) {
								float[] reverseA = new float[a.length];
								for (int i = 0; i < a.length; i++) {
									reverseA[i] = a[a.length - i - 1];
								}
								a = reverseA;
							}
							WeightedObservedPoints obs = new WeightedObservedPoints();
							for (int i = 0; i < a.length; i++) {
								obs.add(a[i], b[i]);
							}
							// Instantiate a n-degree polynomial fitter.
							PolynomialCurveFitter fitter = PolynomialCurveFitter
									.create(nCoeff - 1);
							// Retrieve fitted parameters (coefficients of the
							// polynomial function).
							double[] coeff = fitter.fit(obs.toList());
							float[] B = new float[nCoeff];
							for (int i = 0; i < coeff.length; i++) {
								B[i] = (float) coeff[i];
							}

							// limit coefficient
							float maxCoeff = parameters.getCoeffLimit();
							if ((coeff[1] > maxCoeff || coeff[1] < -maxCoeff)
									&& nCoeff == 2) {
								if (coeff[1] > maxCoeff) {
									B[1] = maxCoeff;
								} else if (coeff[1] < -maxCoeff) {
									B[1] = -maxCoeff;
								}
								double suma = Arrays
										.asList(ArrayUtils.toObject(a))
										.stream().mapToDouble(i -> i).sum();
								double sumb = Arrays
										.asList(ArrayUtils.toObject(b))
										.stream().mapToDouble(i -> i).sum();
								B[0] = (float) ((sumb - B[1] * suma) / a.length);

							}

							// evaluate sum square error
							float R = 0f;
							for (int j = 0; j < a.length; j++) {
								R += Math.pow(((a[j] * B[1]) + B[0]) - b[j], 2);
							}

							if (bestR > R) { // found self
								// parameters that
								// less than stored parameter s
								bestR = R;
								// store minimum value of self similarity
								for (int i = 0; i < nCoeff; i++) {
									codeChunk[i] = B[i];
								}
								// set domain index
								codeChunk[nCoeff] = (float) (dbIdx + 1);
								if (rev == 1) {
									codeChunk[nCoeff] = -(float) (dbIdx + 1);
								}

								// set range block size
								codeChunk[nCoeff + 1] = rangeBlockSize;
								// range block size boundary
								codeChunk[nCoeff + 2] = rangeIdx;
								// domain scale
							}
						}
					}
					partProgress.addAndGet(1);
					samplesProgress.addAndGet(rangeBlockSize);
					return codeChunk; // code of each range block
				}
			});
		}
		try {
			initTime = System.currentTimeMillis();
			List<Future<float[]>> futures = executorService
					.invokeAll(rangeTask);
			completeTime = System.currentTimeMillis();
			for (int c = 0; c < futures.size(); c++) {
				Future<float[]> future = futures.get(c);
				// store minimum code value of self similarity
				code[c] = Arrays.asList(ArrayUtils.toObject(future.get()))
						.stream().mapToDouble(i -> i).toArray();
			}
		} catch (InterruptedException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		} catch (ExecutionException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		} finally {
			executorService.shutdown();
		}
		isDone = true;
		return code; // code of each file
	}

	private float[] resample(float[] d) {
		int domainScale = parameters.getDomainScale();
		float[] result = new float[d.length / domainScale];
		for (int i = 0; i < d.length; i += domainScale) {
			for (int j = 0; j < domainScale; j++) {
				result[i / domainScale] = result[i / domainScale] + d[i + j];
			}
			result[i / domainScale] = result[i / domainScale] / domainScale;
		}
		return result;
	}

	public int countProcessedSamples() {
		return samplesProgress.get();
	}

	public int countProcessedParts() {
		return (int) partProgress.get();
	}

	public int getNParts() {
		return nParts;
	}

	public int getNSamples() {
		return nSamples;
	}

	public long time() {
		return completeTime - initTime;
	}

	public boolean isDone() {
		return isDone;
	}

}
