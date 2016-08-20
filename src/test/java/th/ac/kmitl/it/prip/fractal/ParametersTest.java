package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.junit.Test;

import th.ac.kmitl.it.prip.fractal.Parameters.ProcessName;

public class ParametersTest {

	@Test
	public void testParametersSUBAN416_FIXED_PARTITION_RBS8() throws IOException {
		String[] linesInput = new String[] { "processname compress", "testname SUBAN416_FIXED_PARTITION_RBS8",
				"infile F://IFEFSR//SamplesSpeech//ids16.txt", "inpathprefix F://IFEFSR//SamplesSpeech//speech//16//",
				"outdir F://IFEFSR//AudioFC//FC//QR//", "maxprocess 8", "inext raw", "outext mat", "pthresh 0",
				"reportrate 0", "gpu true", "coefflimit 1.2", "skipifexist true", "minr 8", "maxr 8"

		};
		Parameters parameters = new Parameters(linesInput);
		assertEquals(8, parameters.getMaxParallelProcess());
		assertEquals(0, parameters.getThresh(), 1e-8f);
		assertEquals(0, parameters.getProgressReportRate());
		assertEquals(1.2, parameters.getCoeffLimit(), 1e-4f);
		assertEquals(8, parameters.getMinBlockSize());
		assertEquals(8, parameters.getMaxBlockSize());

		assertTrue(ProcessName.COMPRESS == parameters.getProcessName());
		assertTrue("SUBAN416_FIXED_PARTITION_RBS8".equals(parameters.getTestName()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\ids16.txt".equals(parameters.getInfile()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\speech\\16".equals(parameters.getInPathPrefix()));
		assertTrue("F:\\IFEFSR\\AudioFC\\FC\\QR\\SUBAN416_FIXED_PARTITION_RBS8".equals(parameters.getOutdir()));
		assertTrue("raw".equals(parameters.getInExtension()));
		assertTrue("mat".equals(parameters.getOutExtension()));

		assertTrue(parameters.isGpuEnable());
		assertTrue(parameters.isSkipIfExist());
		assertTrue(parameters.isValidParams());
	}

	@Test
	public void testParametersSUBAN48_FIXED_PARTITION_RBS15() throws IOException {
		String[] linesInput = new String[] { "processname compress", "testname SUBAN48_FIXED_PARTITION_RBS15",
				"infile F://IFEFSR//SamplesSpeech//ids8.txt", "inpathprefix F://IFEFSR//SamplesSpeech//speech//8//",
				"outdir F://IFEFSR//AudioFC//FC//QR//", "maxprocess 7", "inext raw", "outext mat", "pthresh 0",
				"reportrate 0", "gpu true", "coefflimit 1.2", "skipifexist true", "minr 16", "maxr 16"

		};
		Parameters parameters = new Parameters(linesInput);
		assertEquals(7, parameters.getMaxParallelProcess());
		assertEquals(0, parameters.getThresh(), 1e-8f);
		assertEquals(0, parameters.getProgressReportRate());
		assertEquals(1.2, parameters.getCoeffLimit(), 1e-4f);
		assertEquals(16, parameters.getMinBlockSize());
		assertEquals(16, parameters.getMaxBlockSize());

		assertTrue(ProcessName.COMPRESS == parameters.getProcessName());
		assertTrue("SUBAN48_FIXED_PARTITION_RBS15".equals(parameters.getTestName()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\ids8.txt".equals(parameters.getInfile()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\speech\\8".equals(parameters.getInPathPrefix()));
		assertTrue("F:\\IFEFSR\\AudioFC\\FC\\QR\\SUBAN48_FIXED_PARTITION_RBS15".equals(parameters.getOutdir()));
		assertTrue("raw".equals(parameters.getInExtension()));
		assertTrue("mat".equals(parameters.getOutExtension()));

		assertTrue(parameters.isGpuEnable());
		assertTrue(parameters.isSkipIfExist());
		assertTrue(parameters.isValidParams());
	}

	@Test
	public void testParametersSUBSYNTH48_FIXED_PARTITION_RBS8() throws IOException {
		String[] linesInput = new String[] { "processname compress", "processname compress",
				"testname SUBSYNTH48_FIXED_PARTITION_RBS8", "infile F://IFEFSR//SamplesSpeech//synth_ids8.txt",
				"inpathprefix F://IFEFSR//SamplesSpeech//synth//8//", "outdir F://IFEFSR//AudioFC//FC//QR//",
				"maxprocess 7", "inext raw", "outext mat", "pthresh 0", "reportrate 0", "gpu true", "coefflimit 0.99",
				"skipifexist true", "minr 8", "maxr 8"

		};
		Parameters parameters = new Parameters(linesInput);
		assertEquals(7, parameters.getMaxParallelProcess());
		assertEquals(0, parameters.getThresh(), 1e-8f);
		assertEquals(0, parameters.getProgressReportRate());
		assertEquals(0.99, parameters.getCoeffLimit(), 1e-4f);
		assertEquals(8, parameters.getMinBlockSize());
		assertEquals(8, parameters.getMaxBlockSize());

		assertTrue(ProcessName.COMPRESS == parameters.getProcessName());
		assertTrue("SUBSYNTH48_FIXED_PARTITION_RBS8".equals(parameters.getTestName()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\synth_ids8.txt".equals(parameters.getInfile()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\synth\\8".equals(parameters.getInPathPrefix()));
		assertTrue("F:\\IFEFSR\\AudioFC\\FC\\QR\\SUBSYNTH48_FIXED_PARTITION_RBS8".equals(parameters.getOutdir()));
		assertTrue("raw".equals(parameters.getInExtension()));
		assertTrue("mat".equals(parameters.getOutExtension()));

		assertTrue(parameters.isGpuEnable());
		assertTrue(parameters.isSkipIfExist());
		assertTrue(parameters.isValidParams());
	}

	@Test
	public void testInvalidParametersSUBSYNTH48_FIXED_PARTITION_RBS8() throws IOException {
		String[] linesInput = new String[] { "processname compress", "processname compress",
				"testname SUBSYNTH48_FIXED_PARTITION_RBS8", "inpathprefix F://IFEFSR//SamplesSpeech//synth//8//",
				"outdir F://IFEFSR//AudioFC//FC//QR//", "maxprocess 7", "inext raw", "outext mat", "pthresh 0",
				"reportrate 0", "gpu true", "coefflimit 0.99", "skipifexist true", "minr 8", "maxr 8"

		};
		Parameters parameters = new Parameters(linesInput);
		assertEquals(7, parameters.getMaxParallelProcess());
		assertEquals(0, parameters.getThresh(), 1e-8f);
		assertEquals(0, parameters.getProgressReportRate());
		assertEquals(0.99, parameters.getCoeffLimit(), 1e-4f);
		assertEquals(8, parameters.getMinBlockSize());
		assertEquals(8, parameters.getMaxBlockSize());
		assertEquals(null, parameters.getInfile());

		assertEquals(ProcessName.COMPRESS, parameters.getProcessName());
		assertTrue("SUBSYNTH48_FIXED_PARTITION_RBS8".equals(parameters.getTestName()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\synth\\8".equals(parameters.getInPathPrefix()));
		assertTrue("raw".equals(parameters.getInExtension()));
		assertTrue("mat".equals(parameters.getOutExtension()));

		assertTrue(parameters.isGpuEnable());
		assertTrue(parameters.isSkipIfExist());
		assertFalse(parameters.isValidParams());
	}

	@Test
	public void testSetParameter() throws IOException {
		String[] linesInput = new String[] { "processname decompress", "testname SUBSYNTH416_FIXED_PARTITION_RBS8",
				"infile F://IFEFSR//SamplesSpeech//synth_ids16.txt",
				"inpathprefix F://IFEFSR//SamplesSpeech//synth//16//", "outdir F://IFEFSR//AudioFC//FC//QR//",
				"maxprocess 1", "inext raw", "outext mat", "pthresh 1e-4", "reportrate 1000", "gpu false",
				"coefflimit 0.99", "skipifexist false", "minr 8", "maxr 16"

		};
		Parameters parameters = new Parameters(new String[] { "processname decompress" });
		parameters.generateFrom(linesInput);
		assertEquals(1, parameters.getMaxParallelProcess());
		assertEquals(1e-4, parameters.getThresh(), 1e-5f);
		assertEquals(1000, parameters.getProgressReportRate());
		assertEquals(0.99, parameters.getCoeffLimit(), 1e-4f);
		assertEquals(8, parameters.getMinBlockSize());
		assertEquals(16, parameters.getMaxBlockSize());

		assertTrue(ProcessName.DECOMPRESS == parameters.getProcessName());
		assertTrue("SUBSYNTH416_FIXED_PARTITION_RBS8".equals(parameters.getTestName()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\synth_ids16.txt".equals(parameters.getInfile()));
		assertTrue("F:\\IFEFSR\\SamplesSpeech\\synth\\16".equals(parameters.getInPathPrefix()));
		assertTrue("F:\\IFEFSR\\AudioFC\\FC\\QR\\SUBSYNTH416_FIXED_PARTITION_RBS8".equals(parameters.getOutdir()));
		assertTrue("raw".equals(parameters.getInExtension()));
		assertTrue("mat".equals(parameters.getOutExtension()));

		assertFalse(parameters.isGpuEnable());
		assertFalse(parameters.isSkipIfExist());
		assertTrue(parameters.isValidParams());
	}

}
