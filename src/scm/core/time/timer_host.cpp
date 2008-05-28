
#include "timer_host.h"

#include <scm/core/time/timer_interface.h>

namespace scm {
namespace time {

timer_host::timer_host()
{
}

timer_host::~timer_host()
{
}

void
timer_host::add_timer(const std::string&    name,
                      const std::string&    description,
                      const timer_ptr&      timer)
{
}

bool
timer_host::remove_timer(const std::string& name)
{
    return (false);
}

const timer_host::timer_ptr&
timer_host::get_timer(const std::string&    name) const
{
    return (timer_ptr());
}

const timer_host::timer_ptr&
timer_host::operator[](const std::string&   name) const
{
    return (timer_ptr());
}

timer_host::const_iterator
timer_host::begin() const
{
    return (_timers.begin());
}

timer_host::const_iterator
timer_host::end() const
{
    return (_timers.end());
}


} // namespace time
} // namespace scm
