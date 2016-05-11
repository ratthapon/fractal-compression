package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Test;

public class DecompressTest {
//	@Test
	public void T3_ARMS_REC_UTEST_DECOMPRESS() {
		try {
			List<String> inputParams = Files
					.readAllLines(Paths.get(MainExecuterTest.class.getResource(
							"T3_ARMS_REC_UTEST_DECOMPRESS_PARAMS.txt").toURI()));
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
	public void T4_AN4_16K_UTEST_DECOMPRESS() {
		try {
			List<String> inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T4_AN4_16K_UTEST_DECOMPRESS_PARAMS.txt").toURI()));
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

//	@Test
	public void T7_ARMS_REC_UTEST_DECOMPRESS() {
		try {
			List<String> inputParams = Files
					.readAllLines(Paths.get(MainExecuterTest.class.getResource(
							"T7_ARMS_REC_UTEST_DECOMPRESS_PARAMS.txt").toURI()));
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

//	@Test
	public void T8_AN4_16K_UTEST_DECOMPRESS() {
		try {
			List<String> inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T8_AN4_16K_UTEST_DECOMPRESS_PARAMS.txt").toURI()));
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
