package th.ac.kmitl.it.prip.fractal.tuneparams;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import th.ac.kmitl.it.prip.fractal.Executer;
import th.ac.kmitl.it.prip.fractal.MainExecuter;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class Params8KEncode {
	@Test
	public void case0() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T0_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case1() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T1_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case2() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T2_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case3() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T3_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case4() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T4_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case5() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T5_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case6() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T6_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case7() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T7_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case8() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T8_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
	public void case9() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KEncode.class
					.getResource("T9_TUNE_AN4_8K_COMPRESS.txt").toURI()));
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
