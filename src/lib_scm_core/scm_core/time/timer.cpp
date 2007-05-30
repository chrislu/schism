
#include "timer.h"

#include <scm_core/time/detail/get_time.h>

using namespace scm::time;

timer::timer()
    : _start_time(0.0),
      _diff_time(0.0)
{
}

timer::~timer()
{
}

void timer::start()
{
    _start_time = scm::time::detail::get_time();
}

void timer::stop()
{
    _diff_time = scm::time::detail::get_time() - _start_time;
}
