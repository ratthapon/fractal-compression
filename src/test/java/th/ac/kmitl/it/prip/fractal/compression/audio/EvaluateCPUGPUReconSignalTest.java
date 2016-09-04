package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;

import javax.sound.sampled.UnsupportedAudioFileException;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.compression.audio.gpu.CUCompressor;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;
import th.ac.kmitl.it.prip.fractal.decompression.audio.Decompressor;

@RunWith(Parameterized.class)
public class EvaluateCPUGPUReconSignalTest {
	private final int input;

	@Parameterized.Parameters
	public static Collection<Object[]> codeSamplesMap() {
		return Arrays.asList(new Object[][] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 } });
	}

	public EvaluateCPUGPUReconSignalTest(int input) {
		this.input = input;
	}

	@Ignore("Not finish")
	@Test
	public void testSameRecon()
			throws IOException, UnsupportedAudioFileException, InterruptedException, ExecutionException {
		// read co-parameters
		String fileDir = "test-classes//expected//synth_wav//";
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);

		// Set GPU compression params
		Parameters gpuParams = new Parameters(params);
		gpuParams.setParameter("inpathprefix", fileDir);
		gpuParams.setParameter("inext", "raw");
		gpuParams.setParameter("gpu", "true");
		DataSetManager gpuDataSetMngr = new DataSetManager(gpuParams);

		// Encode using GPU
		float[] originSignal = gpuDataSetMngr.readAudio(input);
		CUCompressor gpuCompressor = new CUCompressor(originSignal, gpuParams);
		double[][] gpuCodes = gpuCompressor.process();

		// Decode from GPU fractal codes
		Decompressor gpuDecompressor = new Decompressor(gpuCodes, gpuParams);
		gpuParams.setParameter("processname", "decompress");
		gpuParams.setParameter("fs", "8000");
		double[] gpuReconSignal = gpuDecompressor.decompress();

		System.out.println("PSNR " + PSNR(gpuReconSignal, originSignal));
		Assert.assertTrue("High psnr", PSNR(gpuReconSignal, originSignal) > 80f);

		// Set CPU compression params
		Parameters cpuParams = new Parameters(params);
		cpuParams.setParameter("inpathprefix", fileDir);
		cpuParams.setParameter("inext", "raw");
		cpuParams.setParameter("gpu", "false");

		// Encode using GPU
		Compressor cuCompressor = new Compressor(originSignal, cpuParams);
		double[][] cpuCodes = cuCompressor.process();

		// Decode from GPU fractal codes
		Decompressor cpuDecompressor = new Decompressor(cpuCodes, cpuParams);
		cpuParams.setParameter("processname", "decompress");
		cpuParams.setParameter("fs", "8000");
		double[] cpuReconSignal = cpuDecompressor.decompress();

		System.out.println("PSNR " + PSNR(cpuReconSignal, originSignal));
		Assert.assertTrue("High psnr", PSNR(cpuReconSignal, originSignal) > 80f);
	}

	@Ignore("Not finish yet")
	@Test
	public void testCPUCodingReconPSNR()
			throws IOException, UnsupportedAudioFileException, InterruptedException, ExecutionException {
		// read co-parameters
		String fileDir = "test-classes//expected//synth_wav//";
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);

		// Set CPU compression params
		Parameters cpuParams = new Parameters(params);
		cpuParams.setParameter("inpathprefix", fileDir);
		cpuParams.setParameter("inext", "raw");
		cpuParams.setParameter("gpu", "false");
		DataSetManager cpuDataSetMngr = new DataSetManager(cpuParams);

		// Encode using GPU
		float[] originSignal = cpuDataSetMngr.readAudio(input);
		Compressor cuCompressor = new Compressor(originSignal, cpuParams);
		double[][] cpuCodes = cuCompressor.process();

		// Decode from GPU fractal codes
		Decompressor cpuDecompressor = new Decompressor(cpuCodes, cpuParams);
		cpuParams.setParameter("processname", "decompress");
		cpuParams.setParameter("fs", "8000");
		double[] cpuReconSignal = cpuDecompressor.decompress();

		System.out.println("PSNR " + PSNR(cpuReconSignal, originSignal));
		Assert.assertTrue("High psnr", PSNR(cpuReconSignal, originSignal) > 50f);
	}

	@Ignore("Not finish yet")
	@Test
	public void testGPUCodingReconPSNR()
			throws IOException, UnsupportedAudioFileException, InterruptedException, ExecutionException {
		// read co-parameters
		String fileDir = "test-classes//expected//synth_wav//";
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);

		// Set GPU compression params
		Parameters gpuParams = new Parameters(params);
		gpuParams.setParameter("inpathprefix", fileDir);
		gpuParams.setParameter("inext", "raw");
		gpuParams.setParameter("gpu", "true");
		DataSetManager gpuDataSetMngr = new DataSetManager(gpuParams);

		// Encode using GPU
		float[] originSignal = gpuDataSetMngr.readAudio(input);
		CUCompressor gpuCompressor = new CUCompressor(originSignal, gpuParams);
		double[][] gpuCodes = gpuCompressor.process();

		// Decode from GPU fractal codes
		Decompressor gpuDecompressor = new Decompressor(gpuCodes, gpuParams);
		gpuParams.setParameter("processname", "decompress");
		gpuParams.setParameter("fs", "8000");
		double[] gpuReconSignal = gpuDecompressor.decompress();

		System.out.println("PSNR " + PSNR(gpuReconSignal, originSignal));
		Assert.assertTrue("High psnr", PSNR(gpuReconSignal, originSignal) > 50f);
	}

	private float PSNR(double[] sig1, float[] sig2) {
		final float max_possible = (float) Math.pow(2, 15);
		float psnr = 0.0f;
		int minNSample = Math.min(sig1.length, sig2.length);
		float mse = 0.0f;
		for (int i = 0; i < minNSample; i++) {
			mse += Math.pow(sig1[i] - sig2[i], 2);
		}
		mse = mse / minNSample;
		float sqrtMse = (float) Math.sqrt(mse);
		psnr = (float) (20 * Math.log10(max_possible / sqrtMse));
		return psnr;
	}

}
