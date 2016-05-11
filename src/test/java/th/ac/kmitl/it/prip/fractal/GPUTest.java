package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Test;

public class GPUTest {
//	@Test
	public void T10_ARMS_REC_UTEST_COMPRESS() {
		try {
			new Executer();
			List<String> inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T10_ARMS_REC_UTEST_COMPRESS_PARAMS.txt").toURI()));
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
	public void T1_ARMS_REC_UTEST_COMPRESS() {
		try {
			new Executer();
			List<String> inputParams = Files
					.readAllLines(Paths
							.get(MainExecuterTest.class.getResource(
									"T10.1_ARMS_REC_UTEST_COMPRESS_PARAMS.txt")
									.toURI()));
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
