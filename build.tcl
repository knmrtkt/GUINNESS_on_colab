set project_directory   [file dirname [info script]]
set project_name        "vivado_hls"

cd $project_directory
open_project $project_name

set_top BinCNN
add_files [glob HLS/*.cpp]
add_files [glob HLS/*.txt]
add_files -tb HLS/main.cpp -cflags "-DC_SIMULATION"

open_solution "soulution1"

set_part {xc7a100tcsg324-1}

set main [open HLS/main.cpp a+]
seek $main 0 start
puts $main "/*"
close $main

set cnn [open HLS/cnn.cpp a+]
seek $main 0 start
puts $cnn "/*"
close $cnn

csim_design -O

file copy -force $project_name/soulution1/csim/build/weight.h HLS/weight.h

set main [open HLS/main.cpp a+]
seek $main 0 start
puts $main "#define NO_SETUP_MEM /*"
close $main

set cnn [open HLS/cnn.cpp a+]
seek $main 0 start
puts $cnn "#define NO_SETUP_MEM /*"
close $cnn

csim_design -O
csynth_design

file copy -force $project_name/soulution1/syn/verilog guinness