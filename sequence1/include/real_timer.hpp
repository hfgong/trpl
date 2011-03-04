#ifndef __REAL_TIMER_HPP_INCLUDED_
#define __REAL_TIMER_HPP_INCLUDED_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "cvpr_stub.hpp"

BEGIN_NAMESPACE_CVPR


struct real_timer_t {
    boost::posix_time::ptime start;
    real_timer_t() {
	start = boost::posix_time::microsec_clock::local_time();
    }
    void restart() {
	start = boost::posix_time::microsec_clock::local_time();
    }
    boost::posix_time::time_duration elapsed_pt() { // posix_time
	boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();
	return end - start;
    }
    int elapsed() { // milliseconds
	boost::posix_time::time_duration dt = elapsed_pt();
	return dt.total_milliseconds();
    }
};

template <class Int>
void time_of_day(Int& hours, Int& min, Int& sec, Int& msec)
{

    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration td = now.time_of_day();
    long resolution  = boost::posix_time::time_duration::ticks_per_second();
    long fracsecs = td.fractional_seconds();
    int usecs = 0;
    if (resolution > 1000000)
	usecs =  fracsecs / (resolution / 1000000);
    else
	usecs =  fracsecs * (1000000 / resolution);
    hours = td.hours();
    min = td.minutes();
    sec = td.seconds();
    msec = (usecs/1000);
}

END_NAMESPACE_CVPR


#endif
