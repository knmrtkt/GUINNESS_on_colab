# GUINNESS_on_colab

This repository is a modification of GUINNESS for use with Google colablatory.

It is also possible to use in a local environment.

There is a Jupiter notebook for road tracking that you can easily try by opening it in "Google colablatory".

The main changes are:
- Support for the latest chainer
- Eliminate of GUI function
- Addition of TCL script for high-level synthesis

Changes from the original source code are written in diff.txt.

### 1. How to use 

1. Access the below link
    - [Open road tracking with Colab](https://colab.research.google.com/github/knmrtkt/GUINNESS_on_colab/blob/master/on_colab.ipynb)
1. Select GPU Runtime from the pull-down menu at the top.
1. Execute each cell in order from the top.
    - The variables enclosed in # in each cell are the parameters.
    - Executing the last cell will download `Project_BNN.zip`.
1. Copy the downloaded file to the computer where Vivado HLS is installed.
1. Execute the following commands to execute high-level synthesis.
    ```
    $ unzip Project_BNN.zip
    $ cd content/GUINNESS_on_colab/Project_BNN
    $ vivado_hls build.tcl
    ```
    - When high-level synthesis is completed, the Verilog file and parameter file are output to the `guinness` directory.



The original GUINNESS is here https://github.com/HirokiNakahara/GUINNESS.
