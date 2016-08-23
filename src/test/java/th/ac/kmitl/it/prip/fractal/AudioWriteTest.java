package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

@RunWith(Parameterized.class)
public class AudioWriteTest {

	@Rule
	public TemporaryFolder testFolder = new TemporaryFolder();

	private String input;

	@Parameterized.Parameters
	public static Collection<Object[]> setBooleanParameters() {
		return Arrays.asList(new Object[][] { { "testAudioWrite" } });
	}

	public AudioWriteTest(String input) {
		this.input = input;
	}

	@Test
	public void testRawAudioWrite() throws IOException {
		File tempFolder = testFolder.newFolder();
		DataSetManager.writeaudio(tempFolder.getPath() + "\\" + input, new double[] { 123 }, "raw");
		File audioFile = new File(tempFolder.getPath() + "\\" + input + ".raw");
		assertTrue(audioFile.exists());
	}

	@Test
	public void testWavAudioWrite() throws IOException {
		int fs = 16000;
		File tempFolder = testFolder.newFolder();
		DataSetManager.writeaudio(tempFolder.getPath() + "\\" + input, new double[] { 123 }, "wav", fs);
		File audioFile = new File(tempFolder.getPath() + "\\" + input + ".wav");
		assertTrue(audioFile.exists());
	}

}
