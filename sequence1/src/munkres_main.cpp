/*
 *   Copyright (c) 2007 John Weaver
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

/*
 * Some example code.
 *
 */


//#include <cstdlib>
//#include <ctime>

#include "munkres.hpp"

#include <iostream>

using namespace boost::numeric::ublas;

int main(int argc, char *argv[]) {
    int nrows = 501;
    int ncols = 501;
	
    if ( argc == 3 ) {
	nrows = atoi(argv[1]);
	ncols = atoi(argv[2]);
    }
	
    matrix<double> wmatrix(nrows, ncols);
	
    //srandom(time(NULL)); // Seed random number generator.

    // Initialize matrix with random values.
    for ( int row = 0 ; row < nrows ; row++ ) {
	for ( int col = 0 ; col < ncols ; col++ ) {
	    wmatrix(row,col) = std::abs(row+col-7);//(double)random();
	}
    }

    //wmatrix = trans(wmatrix);

    // Display begin matrix state.
    for ( int row = 0 ; row < nrows ; row++ ) {
	for ( int col = 0 ; col < ncols ; col++ ) {
	    std::cout.width(2);
	    std::cout << wmatrix(row,col) << ",";
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;

    // Apply Munkres algorithm to matrix.
    matrix<double> oldw = wmatrix;
    Munkres<> m;
    m.solve(wmatrix);

    // Display solved matrix.
    for ( int row = 0 ; row < nrows ; row++ ) {
	for ( int col = 0 ; col < ncols ; col++ ) {
	    std::cout.width(2);
	    std::cout << wmatrix(row,col) << ",";
	}
	std::cout << std::endl;
    }

    std::cout << std::endl;

    double s = 0;
    for ( int row = 0 ; row < nrows ; row++ ) {
	for ( int col = 0 ; col < ncols ; col++  ) {
	    if ( wmatrix(row,col) == 0 ) s += oldw(row, col);
	}
    }
    std::cout <<"s="<<s<< std::endl;


	
    for ( int row = 0 ; row < nrows ; row++ ) {
	int rowcount = 0;
	for ( int col = 0 ; col < ncols ; col++  ) {
	    if ( wmatrix(row,col) == 0 )
		rowcount++;
	}
	if ( rowcount != 1 )
	    std::cerr << "Row " << row << " has " << rowcount << " columns that have been matched." << std::endl;
    }

    for ( int col = 0 ; col < ncols ; col++ ) {
	int colcount = 0;
	for ( int row = 0 ; row < nrows ; row++ ) {
	    if ( wmatrix(row,col) == 0 )
		colcount++;
	}
	if ( colcount != 1 )
	    std::cerr << "Column " << col << " has " << colcount << " rows that have been matched." << std::endl;
    }

    return 0;
}
