


-- Sun Aug 22 17:23:25 EDT 2010

This directory contains code for ./bin/bbseg.

bbseg.hpp
	The code does the real things.

bbseg_main.cpp
	The code call bbseg.hpp for each frame and each camera.
	Main function for ./bin/bbseg is here.

bbseg_test.cpp
	The code for ./bin/bbseg_test.
	Because ./bin/bbseg is designed for mpi, this is a non-mpi one frame version for debug purpose.
	This file also contains some sandbox code.

