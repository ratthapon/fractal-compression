# README.md

# Java Fractal Audio Compression

A hybrid Java-CUDA fractal compression framework for audio coding.

This project includes the CUDA GPU computation to faster compress the audio samples to fractal code. This project introduces these features:

- Ready to process by using [MAVEN](https://maven.apache.org/).
- Faster encoding by CUDA devices support from [JCuda](http://www.jcuda.org/).
- Easier audio data processing.
- MATLAB file extension (.mat), and binary format (.bin) are supported for fractal codes.
- RAW (.raw) and .WAV audio file formats are supported.
- Parallel processing is supported.
- Bin-tree partition is supported.
- Parameterized parition is supported.
- Batch processing is supported.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
**Prerequisites**
Install these software for building and launching.

- [Java jdk 1.8](https://www.java.com/en/download/)
- [MAVEN](https://maven.apache.org/download.cgi) 
- [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)

The cuda toolkit maybe exclude if you don't need to process using GPU.
**Installing**
Install all aboved prerequisites. Then install by following steps.

1. Clone the project to your local repository.
    git clone https://github.com/ratthapon/java-fractal-audio-compression jfac
2. Compile the ptx files on your machine by exec these commands in cmd.
    cd jfac
    ctx-compile-script-2
3. Install the project using MAVEN.
    mvn install

It will install the program if the tests are passed. You can add *-DskipTests* if you want to install without gpu support.
Check the build message, it should be passed.
**Alternative installation using Eclipse**

1. Clone and built as above, then exec these commands.
    mvn eclipse:eclipse
2. Use the Eclipse to open this project then setup the run configuration by following:
- Run -> Run Configuration -> Java Application -> Main -> Project
    jfac
- Run -> Run Configuration -> Java Application -> Main -> Main Class
    th.ac.kmitl.it.prip.fractal.MainExecuter
- Run -> Run Configuration -> Java Application -> Arguments -> Working directory -> Other
    ${workspace_loc:jfac/target}

Then you can use Eclipse to execute this program.

## Running the tests

Test the program is built properly by using MAVEN

    mvn test
## Deployment

This project can be executed by variety of methods. You can use any one that you prefer.

- Exec by using MAVEN at the cmd,
    mvn ecec:exec

, after the maven built and running the process, give the parameters to cmd and double Enter to run it with given parameters.

- Exec by MAVEN with a pre-descripted parameter file by following commands,
    mvn ecec:exec -Dparam="path_to_param/params.txt"
- Use as a lib, please see API list. [under development]
## Example Parameters

Each parameter is delimited by newline.
The parameter name and value are delimited by a space as follows.

    processname compress
    testname actual_synth
    infile test-classes//synth-file-list.txt
    inpathprefix test-classes//expected//synth_wav//
    outdir test-classes//
    maxprocess 7
    inext raw
    outext mat
    pthresh 0
    reportrate 0
    gpu true
    coefflimit 1.2
    skipifexist false
    minr 4
    maxr 4
    
    

See available parameters list [under development].

## Built With
- [MAVEN](https://maven.apache.org/) - Dependency Management
## Fractal compression

Fractal compression is a data compression algorithm. It compress the raw data into smaller "code". There are many schemes to implement this algorithm but the fractal block coding is the most popular scheme. Fractal block coding store only a self-similarity parameters of each partitioned block. It is so called fractal code. The self-similarity parameters is used to compose the original data from itself.

## Contributing

All contribution are gracefully accept.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ratthapon/fractal-compression/tags).

## Authors
- **Rattaphon Hokking** - *Initial work* - [ratthapon](https://github.com/ratthapon)

See also the list of [contributors](https://github.com/ratthapon/fractal-compression/graphs/contributors) who participated in this project.

## Acknowledgments
- Thanks to [jmatio](https://github.com/gradusnikov/jmatio) 
- Thanks to [Apache Commons](https://commons.apache.org/)
- Thanks to [JCUDA](http://www.jcuda.org/) 
- Thanks to [PRIP lab](http://prip.it.kmitl.ac.th/) for their resources to develop this project
## Issues
- CUDA device maybe error while run a long batching.
- Codes of GPU and CPU compression mismatch (GPU compression uses QR decomposition while CPU compression use Levenberg–Marquardt algorithm in least square optimization process).
- Repository's merged commits are lost.
## References
1. M. F. Barnsley and L. P. Hurd, *Fractal image compression*. AK Peters, Ldt., 1993.
2. A. E. Jacquin, “[Fractal image coding: a review](http://ieeexplore.ieee.org/document/241507/),” *Proceedings of the IEEE*, vol. 81, no. 10, pp. 1451–1465, Oct. 1993.
3. M. F. Barnsley, [*Fractals Everywhere: New Edition*](http://store.doverpublications.com/0486488705.html), New edition edition. Dover Publications, 2013.
4. Y. Fisher, [](http://www.springer.com/gp/book/9781461275527)[*Fractal image compression: theory and application*](http://www.springer.com/gp/book/9781461275527). Springer Science & Business Media, 2012.
5. R. Hokking, K. Woraratpanya, and Y. Kuroki, “[Speech recognition of different sampling rates using fractal code descriptor](http://ieeexplore.ieee.org/document/7748895/),” in *2016 13th International Joint Conference on Computer Science and Software Engineering (JCSSE)*, 2016, pp. 1–5.
[](http://ieeexplore.ieee.org/document/7748895/)## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/ratthapon/fractal-compression/blob/master/LICENSE.txt) file for details.
