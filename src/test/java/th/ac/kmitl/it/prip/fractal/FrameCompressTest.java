package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Test;

public class FrameCompressTest {
	@Test
	public void T5_ARMS_REC_UTEST_COMPRESS() {
		try {
			List<String> inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T5_ARMS_REC_UTEST_COMPRESS_PARAMS.txt").toURI()));
			Executer.processParameters(inputParams);
			MainExecuter.exec();
		} catch (IOException e) {
			e.printStackTrace();
			fail("IOEXception");
		} catch (URISyntaxException e) {
			fail("URISyntaxException");
			e.printStackTrace();
		}
	}

	@Test
	public void T6_AN4_16K_UTEST_COMPRESS() {
		try {
			List<String> inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T6_AN4_16K_UTEST_COMPRESS_PARAMS.txt").toURI()));
			Executer.processParameters(inputParams);
			MainExecuter.exec();
		} catch (IOException e) {
			e.printStackTrace();
			fail("IOEXception");
		} catch (URISyntaxException e) {
			fail("URISyntaxException");
			e.printStackTrace();
		}
	}
}
