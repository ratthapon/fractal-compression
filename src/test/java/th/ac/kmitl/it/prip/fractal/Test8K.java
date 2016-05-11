package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class Test8K {
	@Test
	public void compress() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths
					.get(MainExecuterTest.class.getResource(
							"T2_AN4_8K_UTEST_COMPRESS_PARAMS.txt").toURI()));
			Executer.processParameters(inputParams);
			MainExecuter.exec();

			inputParams = Files.readAllLines(Paths.get(MainExecuterTest.class
					.getResource("T2_AN4_16K_UTEST_COMPRESS_PARAMS.txt")
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

	@Test
	public void decompress() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(MainExecuterTest.class
					.getResource("T4_AN4_8K_UTEST_DECOMPRESS_PARAMS.txt")
					.toURI()));
			Executer.processParameters(inputParams);
			MainExecuter.exec();

			inputParams = Files.readAllLines(Paths.get(MainExecuterTest.class
					.getResource("T4_AN4_16K_UTEST_DECOMPRESS_PARAMS.txt")
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
