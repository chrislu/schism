
#include "time_system.h"

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>

#include <scm_core/console.h>
#include <scm_core/time/detail/highres_time_stamp.h>

using namespace scm::time;

time_system::time_system()
{
}

time_system::~time_system()
{
}

bool time_system::initialize()
{
    _high_res_timer.reset(new detail::high_res_time_stamp());

    if (!_high_res_timer->initialize()) {
        console.get() << con::log_level(con::error)
                      << "time_system::initialize(): "
                      << "unable to initialize high resolution timer" << std::endl;
        return (false);
    }

    return (true);
}

bool time_system::shutdown()
{
    return (true);
}

ptime time_system::get_local_time()
{
    return (boost::posix_time::microsec_clock::local_time());
}

ptime time_system::get_universal_time()
{
    return (boost::posix_time::microsec_clock::universal_time());
}

date time_system::get_local_date()
{
    return (boost::gregorian::day_clock::local_day());
}

date time_system::get_universal_date()
{
    return (boost::gregorian::day_clock::universal_day());
}

inline time_stamp time_system::get_time_stamp()
{
    return (detail::high_res_time_stamp::now());
}

inline time_duration time_system::get_elapsed_duration(time_stamp start,
                                                       time_stamp end)
{
    // look to wrap arounds
    time_stamp  diff((end > start) ? (end - start) : (~start + 1 + end));

    double      dur =   static_cast<double>(diff)
                      / static_cast<double>(detail::high_res_time_stamp::ticks_per_second());
    
    return (nanosec(static_cast<time_stamp>(dur * 1e9)));
}
