
#include "timer_interface.h"

namespace scm {
namespace time {

timer_interface::timer_interface(resolution_type res_type)
  : _duration(0, 0, 0, 0),
    _res_type(res_type)
{
}
timer_interface::~timer_interface()
{
}

timer_interface::duration_type
timer_interface::get_time() const
{
    collect_result();
    return (_duration);
}

timer_interface::resolution_type
timer_interface::resolution() const
{
    return (_res_type);
}

} // namespace time
} // namespace scm
