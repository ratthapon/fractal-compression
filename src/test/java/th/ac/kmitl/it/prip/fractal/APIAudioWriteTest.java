package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

@RunWith(Parameterized.class)
public class APIAudioWriteTest {

	@Rule
	public TemporaryFolder testFolder = new TemporaryFolder();

	private int input;
	private String output;

	@Parameterized.Parameters
	public static Collection<Object[]> setBooleanParameters() {
		return Arrays.asList(new Object[][] { { 0, "synth-8" } });
	}

	public APIAudioWriteTest(int input, String output) {
		this.input = input;
		this.output = output;
	}

	@Test
	public void testRawAudioWriteAPI() throws IOException {
		File tempFolder = testFolder.newFolder();
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);
		Parameters testParameters = new Parameters(params);
		testParameters.setParameter("outdir", Paths.get(tempFolder.getPath()).toString());
		testParameters.setParameter("outext", "raw");
		DataSetManager dataSetManager = new DataSetManager(testParameters);

		dataSetManager.writeAudio(input, new double[] { 123 });

		File audioFile = new File(tempFolder.getPath() + "\\" + output + ".raw");
		assertTrue(audioFile.exists());
	}

	@Test
	public void testWavAudioWriteAPI() throws IOException {
		int fs = 16000;
		File tempFolder = testFolder.newFolder();
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);
		Parameters testParameters = new Parameters(params);
		testParameters.setParameter("outdir", Paths.get(tempFolder.getPath()).toString());
		testParameters.setParameter("outext", "wav");
		testParameters.setParameter("fs", "" + fs);
		DataSetManager dataSetManager = new DataSetManager(testParameters);

		dataSetManager.writeAudio(input, new double[] { 123 });

		File audioFile = new File(tempFolder.getPath() + "\\" + output + ".wav");
		assertTrue(audioFile.exists());
	}

}
