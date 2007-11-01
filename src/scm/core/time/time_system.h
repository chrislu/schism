
#ifndef TIME_SYSTEM_H_INCLUDED
#define TIME_SYSTEM_H_INCLUDED

#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <scm/core/sys_interfaces.h>
#include <scm/core/ptr_types.h>
#include <scm/core/time/time_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace time {

namespace detail {

class high_res_time_stamp;

} // namespace detail

class __scm_export(core) time_system : public scm::core::system
{
public:
    time_system();
    virtual ~time_system();

    // core::system interface
    bool                                initialize();
    bool                                shutdown();

    static ptime                        get_local_time();
    static ptime                        get_universal_time();

    static date                         get_local_date();
    static date                         get_universal_date();

    static time_stamp                   get_time_stamp(); 

    static time_duration                get_elapsed_duration(time_stamp /*start*/,
                                                             time_stamp /*end*/);




private:
    core::scoped_ptr<detail::high_res_time_stamp>   _high_res_timer;

}; // class time_system

} // namespace time
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TIME_SYSTEM_H_INCLUDED
