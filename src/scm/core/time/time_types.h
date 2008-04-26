
#ifndef TIME_TYPES_H_INCLUDED
#define TIME_TYPES_H_INCLUDED

#include <scm/core/platform/config.h>

#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <scm/core/int_types.h>

namespace scm {
namespace time {

using boost::gregorian::date;
using boost::gregorian::date_duration;

using boost::gregorian::days;
using boost::gregorian::weeks;
using boost::gregorian::months;
using boost::gregorian::years;

using boost::posix_time::ptime;
using boost::posix_time::time_duration;

using boost::posix_time::hours;
using boost::posix_time::minutes;
using boost::posix_time::seconds;
using boost::posix_time::millisec;
using boost::posix_time::microsec;

#ifndef BOOST_DATE_TIME_HAS_NANOSECONDS
typedef boost::date_time::subsecond_duration<boost::posix_time::time_duration, 1000000000> nanosec;
#else
using boost::posix_time::nanosec;
#endif

typedef scm::uint64     time_stamp;

inline double to_seconds(time_duration dur) {
    return (static_cast<double>(dur.total_nanoseconds()) * 0.000000001);
}

inline double to_milliseconds(time_duration dur) {
    return (static_cast<double>(dur.total_nanoseconds()) * 0.000001);
}

inline double to_microseconds(time_duration dur) {
    return (static_cast<double>(dur.total_nanoseconds()) * 0.001);
}

inline double to_nanoseconds(time_duration dur) {
    return (static_cast<double>(dur.total_nanoseconds()));
}

} // namespace time
} // namespace scm

#endif // TIME_TYPES_H_INCLUDED
