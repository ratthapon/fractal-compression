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

@RunWith(Parameterized.class)
public class CodeWriteTest {

	@Rule
	public TemporaryFolder testFolder = new TemporaryFolder();

	private String input;

	@Parameterized.Parameters
	public static Collection<Object[]> setBooleanParameters() {
		return Arrays.asList(new Object[][] { { "testCodeWrite" } });
	}

	public CodeWriteTest(String input) {
		this.input = input;
	}

	@Test
	public void testMatAudioWrite() throws IOException {
		File tempFolder = testFolder.newFolder();
		DataHandler.writecode(tempFolder.getAbsolutePath() + "\\" + input, new double[][] { { 123 } }, "mat");
		File audioFile = new File(tempFolder.getAbsolutePath() + "\\" + input + ".mat");
		assertTrue(audioFile.exists());
	}

	@Test
	public void testBinAudioWrite() throws IOException {
		File tempFolder = testFolder.newFolder();
		DataHandler.writecode(tempFolder.getAbsolutePath() + "\\" + input, new double[][] { { 321 } }, "bin");
		File audioFile = new File(tempFolder.getAbsolutePath() + "\\" + input + ".bin");
		assertTrue(audioFile.exists());
	}

}
