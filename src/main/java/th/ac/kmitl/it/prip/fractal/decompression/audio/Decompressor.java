package th.ac.kmitl.it.prip.fractal.decompression.audio;

import java.util.Arrays;
import java.util.logging.Logger;

import org.apache.commons.lang3.ArrayUtils;

import th.ac.kmitl.it.prip.fractal.Parameters;

public class Decompressor {
	private static final Logger LOGGER = Logger.getLogger(Decompressor.class.getName());

	private Parameters parameters = null;
	private double[][] codes = null;
	private int nParts;
	private int nSamples;
	private long initTime;
	private long completeTime;
	private boolean isDone = false;
	private static int PART_SIZE_ELEMENT;
	private static int DOMAIN_INDEX_ELEMENT;

	public Decompressor(double[][] data, Parameters compressParameters) {
		parameters = (Parameters) compressParameters;
		codes = data;
		nSamples = 0;
		nParts = data.length;
		PART_SIZE_ELEMENT = data[0].length - 2;
		DOMAIN_INDEX_ELEMENT = Math.abs(data[0].length - 3);
		for (int i = 0; i < data.length; i++) {
			nSamples += data[i][PART_SIZE_ELEMENT];
		}
	}

	public double[] decompress() {
		final int nCoeff = parameters.getNCoeff();
		double[] audioData = new double[(int) (nSamples * parameters.getAlpha())];
		double[] bufferAudioData = new double[audioData.length];
		initTime = System.currentTimeMillis();
		// iteration
		for (int iter = 0; iter < parameters.getMaxIteration(); iter++) {
			// each code
			int rbIdx = 0;
			for (int fIdx = 0; fIdx < nParts; fIdx++) {
				int bufferSize = (int) (codes[fIdx][PART_SIZE_ELEMENT] * parameters.getAlpha());
				double[] domain = new double[bufferSize * parameters.getDomainScale()];
				int domainIdx = Math.abs((int) (codes[fIdx][DOMAIN_INDEX_ELEMENT] * parameters.getAlpha())) - 1;
				domain = resample(Arrays.copyOfRange(audioData, domainIdx, domainIdx + domain.length));
				if (codes[fIdx][DOMAIN_INDEX_ELEMENT] < 0) { // if should be reverse
					ArrayUtils.reverse(domain);
				}

				double[] buffer = interpolateBlock(domain, fIdx, nCoeff, bufferSize);
				for (int i = 0; i < buffer.length; i++) {
					bufferAudioData[rbIdx + i] = buffer[i];
				}
				rbIdx += bufferSize;
			}
			if (isInvalidReconstructed(bufferAudioData)) {
				completeTime = System.currentTimeMillis();
				isDone = true;
				return audioData;
			} else {
				audioData = bufferAudioData;
			}
		}
		completeTime = System.currentTimeMillis();
		isDone = true;
		return audioData;
	}

	private double[] interpolateBlock(double[] domain, int fIdx, final int nCoeff, int bufferSize) {
		double[] buffer = new double[bufferSize];
		for (int i = 0; i < buffer.length; i++) {
			for (int coeffOrder = 0; coeffOrder < nCoeff; coeffOrder++) {
				buffer[i] += codes[fIdx][coeffOrder] * Math.pow(domain[i], coeffOrder);
			}
		}
		return buffer;
	}

	private boolean isInvalidReconstructed(double[] bufferAudioData) {
		return ArrayUtils.contains(bufferAudioData, Float.NaN)
				|| ArrayUtils.contains(bufferAudioData, Float.POSITIVE_INFINITY)
				|| ArrayUtils.contains(bufferAudioData, Float.NEGATIVE_INFINITY);
	}

	private double[] resample(double[] domain) {
		int domainScale = parameters.getDomainScale();
		double[] result = new double[domain.length / domainScale];
		for (int i = 0; i < domain.length; i += domainScale) {
			for (int j = 0; j < domainScale; j++) {
				result[i / domainScale] = result[i / domainScale] + domain[i + j];
			}
		}
		for (int i = 0; i < result.length; i++) {
			result[i] = result[i] / domainScale;
		}
		return result;
	}

	public int getNParts() {
		return nParts;
	}

	public int getNSamples() {
		return nSamples;
	}

	public long getInitTime() {
		return initTime;
	}

	public long getCompleteTime() {
		return completeTime;
	}

	public boolean isDone() {
		return isDone;
	}

	public long time() {
		return completeTime - initTime;
	}

}
