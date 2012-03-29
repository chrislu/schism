
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TIME_TYPES_H_INCLUDED
#define TIME_TYPES_H_INCLUDED

#include <scm/core/platform/config.h>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <scm/core/numeric_types.h>

namespace scm {
namespace time {

typedef boost::gregorian::date              date;
typedef boost::gregorian::date_duration     date_duration;

typedef boost::gregorian::days              days;
typedef boost::gregorian::weeks             weeks;
typedef boost::gregorian::months            months;
typedef boost::gregorian::years             years;

typedef boost::posix_time::ptime            ptime;
typedef boost::posix_time::time_duration    time_duration;

typedef boost::posix_time::hours            hours;
typedef boost::posix_time::minutes          minutes;
typedef boost::posix_time::seconds          seconds;
typedef boost::posix_time::millisec         millisec;
typedef boost::posix_time::microsec         microsec;

#ifndef BOOST_DATE_TIME_HAS_NANOSECONDS
    typedef boost::date_time::subsecond_duration<boost::posix_time::time_duration, 1000000000> nanosec;
#else
    typedef boost::posix_time::nanosec      nanosec;
#endif

typedef scm::uint64                         time_stamp;

inline double to_seconds(time_duration dur)
{
    return (static_cast<double>(dur.total_nanoseconds()) * 0.000000001);
}

inline double to_milliseconds(time_duration dur)
{
    return (static_cast<double>(dur.total_nanoseconds()) * 0.000001);
}

inline double to_microseconds(time_duration dur)
{
    return (static_cast<double>(dur.total_nanoseconds()) * 0.001);
}

inline double to_nanoseconds(time_duration dur)
{
    return (static_cast<double>(dur.total_nanoseconds()));
}

} // namespace time
} // namespace scm

#endif // TIME_TYPES_H_INCLUDED
