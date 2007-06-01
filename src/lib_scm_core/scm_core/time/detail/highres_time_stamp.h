
#ifndef HIGHRES_TIME_STAMP_H_INCLUDED
#define HIGHRES_TIME_STAMP_H_INCLUDED

#include <scm_core/time/time_types.h>

namespace scm {
namespace time {
namespace detail {

class high_res_time_stamp
{
public:
    high_res_time_stamp();
    virtual ~high_res_time_stamp();

    bool                        initialize();

    time_duration               get_overhead() const;

    static time_stamp           ticks_per_second();
    static time_stamp           now();

private:
    time_duration               _overhead;

}; // class high_res_time_stamp

} // namespace detail
} // namespace time
} // namespace scm

#endif // HIGHRES_TIME_STAMP_H_INCLUDED
