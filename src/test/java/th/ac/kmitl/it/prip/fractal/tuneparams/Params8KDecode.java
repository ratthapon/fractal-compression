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
public class Params8KDecode {
	
	@Test
	public void dCase0() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T0_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase1() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T1_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase2() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T2_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase3() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T3_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase4() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T4_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase5() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T5_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase6() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T6_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase7() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T7_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase8() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T8_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
	public void dCase9() {
		try {
			List<String> inputParams;
			inputParams = Files.readAllLines(Paths.get(Params8KDecode.class
					.getResource("T9_TUNE_AN4_8K_DECOMPRESS.txt").toURI()));
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
