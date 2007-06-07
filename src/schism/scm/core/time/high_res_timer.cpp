
#include "high_res_timer.h"

#include <scm/core/time/time_system.h>

using namespace scm::time;

high_res_timer::high_res_timer()
    : _start(0)
{
}

high_res_timer::~high_res_timer()
{

}

void high_res_timer::start()
{
    _start = time_system::get_time_stamp();
}

void high_res_timer::stop()
{
    _duration = time_system::get_elapsed_duration(_start, 
                                                  time_system::get_time_stamp());
}
