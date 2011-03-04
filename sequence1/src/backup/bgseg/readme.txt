


-- Mon Sep 13 16:54:01 EDT 2010


This directory contains code for ./bin/bgseg. For background segmentation.

bgseg.hpp
	The code does the real things.

bgseg_main.cpp
	The code call bgseg.hpp
	Main function for ./bin/bgseg is here.

bgseg_test.cpp
	The code for ./bin/bgseg_test.
	Because ./bin/bgseg is designed for mpi, this is a non-mpi one frame version for debug purpose.
	This file also contains some sandbox code.

