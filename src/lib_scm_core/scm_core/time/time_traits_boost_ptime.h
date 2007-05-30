

#ifndef TIME_TRAITS_BOOST_PTIME_H_INCLUDED
#define TIME_TRAITS_BOOST_PTIME_H_INCLUDED

#include <scm_core/time/time_traits.h>

#include <boost/call_traits.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace scm {
namespace core {

template<>
struct time_traits<boost::posix_time::ptime>
{
    typedef boost::posix_time::ptime            time_t;
    typedef boost::posix_time::time_duration    time_diff_t;

    static const time_t         current_time() {
        return (boost::posix_time::microsec_clock::universal_time());
    }

    static const time_diff_t    difference(boost::call_traits<time_t>::param_type lhs,
                                           boost::call_traits<time_t>::param_type rhs) {
        return (lhs - rhs);
    }

    static const bool           less(boost::call_traits<time_t>::param_type lhs,
                                     boost::call_traits<time_t>::param_type rhs) {
        return (lhs < rhs);
    }

    static const double         to_milliseconds(boost::call_traits<time_diff_t>::param_type diff) {
        return (static_cast<double>(diff.total_nanoseconds()) * 0.000001);
    }
}; // struct time_traits<boost::posix_time::ptime>


} // namespace core
} // namespace scm

#endif // TIME_TRAITS_BOOST_PTIME_H_INCLUDED
