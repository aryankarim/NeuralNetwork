# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aileen/projects/NeuralNetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aileen/projects/NeuralNetwork/build

# Include any dependencies generated for this target.
include CMakeFiles/neuralnetwork.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neuralnetwork.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neuralnetwork.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neuralnetwork.dir/flags.make

CMakeFiles/neuralnetwork.dir/main.cpp.o: CMakeFiles/neuralnetwork.dir/flags.make
CMakeFiles/neuralnetwork.dir/main.cpp.o: ../main.cpp
CMakeFiles/neuralnetwork.dir/main.cpp.o: CMakeFiles/neuralnetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aileen/projects/NeuralNetwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neuralnetwork.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralnetwork.dir/main.cpp.o -MF CMakeFiles/neuralnetwork.dir/main.cpp.o.d -o CMakeFiles/neuralnetwork.dir/main.cpp.o -c /home/aileen/projects/NeuralNetwork/main.cpp

CMakeFiles/neuralnetwork.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuralnetwork.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aileen/projects/NeuralNetwork/main.cpp > CMakeFiles/neuralnetwork.dir/main.cpp.i

CMakeFiles/neuralnetwork.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuralnetwork.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aileen/projects/NeuralNetwork/main.cpp -o CMakeFiles/neuralnetwork.dir/main.cpp.s

# Object files for target neuralnetwork
neuralnetwork_OBJECTS = \
"CMakeFiles/neuralnetwork.dir/main.cpp.o"

# External object files for target neuralnetwork
neuralnetwork_EXTERNAL_OBJECTS =

neuralnetwork: CMakeFiles/neuralnetwork.dir/main.cpp.o
neuralnetwork: CMakeFiles/neuralnetwork.dir/build.make
neuralnetwork: CMakeFiles/neuralnetwork.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aileen/projects/NeuralNetwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable neuralnetwork"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neuralnetwork.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neuralnetwork.dir/build: neuralnetwork
.PHONY : CMakeFiles/neuralnetwork.dir/build

CMakeFiles/neuralnetwork.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neuralnetwork.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neuralnetwork.dir/clean

CMakeFiles/neuralnetwork.dir/depend:
	cd /home/aileen/projects/NeuralNetwork/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aileen/projects/NeuralNetwork /home/aileen/projects/NeuralNetwork /home/aileen/projects/NeuralNetwork/build /home/aileen/projects/NeuralNetwork/build /home/aileen/projects/NeuralNetwork/build/CMakeFiles/neuralnetwork.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neuralnetwork.dir/depend

