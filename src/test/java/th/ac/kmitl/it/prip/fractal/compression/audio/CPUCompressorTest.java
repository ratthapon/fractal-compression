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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

@RunWith(Parameterized.class)
public class CPUCompressorTest {
	private int input;
	private double[][] expected;

	@Parameterized.Parameters
	public static Collection<Object[]> codeSamplesMap() {
		return Arrays.asList(new Object[][] {
				{ 1, new double[][] { { 452.3589, -0.7515, 1.0, 4.0, 0 }, { 118.7057, -0.2090, 2.0, 4.0, 1.0 },
						{ 62.2481, -0.0882, 1.0, 4.0, 2.0 }, { 37.0921, -0.0277, 1.0, 4.0, 3.0 },
						{ 29.0183, -0.0245, -2.0, 4.0, 4.0 } } },
				{ 2, new double[][] { { 40.8, -1.2, -2.0, 4.0, 0 }, { 37.3, -1.2, -2.0, 4.0, 1.0 },
						{ 34.8, -1.2, -2.0, 4.0, 2.0 }, { -9.8, 1.2, 2.0, 4.0, 3.0 }, { -11.3, 1.2, 2.0, 4.0, 4.0 },
						{ -10.6, 1.2, 3.0, 4.0, 5.0 }, { -12.1, 1.2, 3.0, 4.0, 6.0 },
						{ 0.5556, 0.8889, 27.0, 4.0, 7.0 }, { -14.3500, 1.2, -3.0, 4.0, 8.0 },
						{ -15.6, 1.2, -3.0, 4.0, 9.0 } } },
				{ 4, new double[][] { { -10.0007, 0.0078, -31.0, 4.0, 0 }, { -8.9436, -0.0303, -36.0, 4.0, 1.0 },
						{ -13.7411, 0.0159, 31.0, 4.0, 2.0 }, { -16.7275, 0.0279, 31.0, 4.0, 3.0 },
						{ -21.4716, 0.0397, 31.0, 4.0, 4.0 }, { -28.0312, 0.0506, 31.0, 4.0, 5.0 },
						{ -41.4010, 0.0605, 31.0, 4.0, 6.0 }, { -78.4428, 0.0762, -33.0, 4.0, 7.0 },
						{ -426.9, 1.2, -34.0, 4.0, 8.0 }, { 106.5, -1.2, -29.0, 4.0, 9.0 },
						{ 79.7912, 0.0549, -33.0, 4.0, 10.0 }, { 48.7213, 0.0467, 31.0, 4.0, 11.0 },
						{ 28.5714, -0.0460, -31.0, 4.0, 12.0 }, { 20.9291, -0.0404, -31.0, 4.0, 13.0 },
						{ 16.7458, -0.0315, -31.0, 4.0, 14.0 }, { 13.3581, -0.0219, -31.0, 4.0, 15.0 },
						{ 11.4804, -0.0121, -31.0, 4.0, 16.0 } } },
				{ 6, new double[][] { { 904.5369, -0.7520, 1.0, 4.0, 0 }, { 233.2513, -0.2005, 1.0, 4.0, 1.0 },
						{ 120.6181, -0.0762, 1.0, 4.0, 2.0 }, { 69.0993, -0.0113, 1.0, 4.0, 3.0 },
						{ 65.7242, -0.0403, -2.0, 4.0, 4.0 } } },
				{ 7, new double[][] { { 86.2500, -1.2, -2.0, 4.0, 0 }, { 80.2500, -1.2, -2.0, 4.0, 1.0 },
						{ -14.2500, 1.2, 2.0, 4.0, 2.0 }, { -17.7500, 1.2, 2.0, 4.0, 3.0 },
						{ -20.5, 1.2, 2.0, 4.0, 4.0 }, { -12.3500, 1.2, 8.0, 4.0, 5.0 },
						{ 0.9463, 0.9008, -21.0, 4.0, 6.0 }, { -16.8500, 1.2, -8.0, 4.0, 7.0 },
						{ -25.2, 1.2, -4.0, 4.0, 8.0 }, { -31.0, 1.2, -2.0, 4.0, 9.0 } } },
				{ 9, new double[][] { { 7.3, -1.2, 10.0, 4.0, 0 }, { 7.0500, -1.2, 10.0, 4.0, 1.0 },
						{ 6.0500, -1.2, 10.0, 4.0, 2.0 }, { 5.3, -1.2, 10.0, 4.0, 3.0 }, { 4.8, -1.2, 10.0, 4.0, 4.0 },
						{ 4.3500, -1.2, -8.0, 4.0, 5.0 }, { -2.8, 1.2, 10.0, 4.0, 6.0 },
						{ -2.7500, 1.2, 14.0, 4.0, 7.0 }, { -3.3, 1.2, 10.0, 4.0, 8.0 }, { -3.3, 1.2, 10.0, 4.0, 9.0 },
						{ -3.8, 1.2, 10.0, 4.0, 10.0 }, { -3.7500, 1.2, 14.0, 4.0, 11.0 },
						{ -4.3, 1.2, 10.0, 4.0, 12.0 }, { 2.0500, -1.2, 10.0, 4.0, 13.0 },
						{ 1.3, -1.2, 10.0, 4.0, 14.0 }, { 0.8, -1.2, 10.0, 4.0, 15.0 },
						{ 0.0500, -1.2, 10.0, 4.0, 16.0 } } },

		});
	}

	public CPUCompressorTest(int input, double[][] expectedResult) {
		this.input = input;
		this.expected = expectedResult;
	}

	@Test
	public void testCompress()
			throws IOException, UnsupportedAudioFileException, InterruptedException, ExecutionException {
		String fileDir = "test-classes//expected//synth_wav//";
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);
		Parameters testParameters = new Parameters(params);
		testParameters.setParameter("inpathprefix", fileDir);
		testParameters.setParameter("inext", "raw");
		testParameters.setParameter("gpu", "false");
		DataSetManager dataSetManager = new DataSetManager(testParameters);

		float[] audioData = dataSetManager.readAudio(input);
		Compressor cuCompressor = new Compressor(audioData, testParameters);
		double[][] actualData = cuCompressor.process();
		float sumAbsError1 = 0f;
		float sumAbsError2 = 0f;
		float sumAbsError3 = 0f;
		for (int i = 0; i < actualData.length; i++) {
			// assert power 0 coefficients maybe different from gpu compression
			// assertEquals(this.expected[i][0], actualData[i][0], 90);
			sumAbsError1 += Math.abs(this.expected[i][0] - actualData[i][0]);

			// assert power 1 coefficients maybe different from gpu compression
			sumAbsError2 += Math.abs(this.expected[i][1] - actualData[i][1]);
			// assertEquals(this.expected[i][1], actualData[i][1], 1e-3);

			// assert domain location maybe different from gpu compression
			for (int j = 2; j < actualData[i].length - 1; j++) {
				sumAbsError3 += Math.abs(this.expected[i][j] - actualData[i][j]);
				// System.out.println("error " + (this.expected[i][j] -
				// actualData[i][j]));
				// assertEquals(Math.abs(this.expected[i][j]),
				// Math.abs(actualData[i][j]), 0);
			}
		}
		float meanAbsError1 = (sumAbsError1 / actualData.length);
		float meanAbsError2 = (sumAbsError2 / actualData.length);
		float meanAbsError3 = (sumAbsError3 / actualData.length);
		Assert.assertTrue("Mismatch bias tolerance", meanAbsError1 < 10);
		Assert.assertTrue("Mismatch first order coeff tolerance", meanAbsError2 < 1);
		Assert.assertTrue("Mismatch dIdx tolerance", meanAbsError3 < 1);
	}

}
