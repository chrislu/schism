
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef HIGH_RES_TIMER_H_INCLUDED
#define HIGH_RES_TIMER_H_INCLUDED

#include <scm/core/time/timer_interface.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) high_res_timer : public timer_interface
{

public:
    high_res_timer(resolution_type res_type = micro_seconds);
    virtual ~high_res_timer();

    void                start();
    void                stop();
    void                intermediate_stop();
    void                collect_result() const;

private:
    time_stamp          _start;

    time_stamp          now() const;
    time_duration       elapsed_duration(time_stamp start,
                                         time_stamp end) const;

}; // class timer_interface

} // namespace time
} // namespace scm

#endif // HIGH_RES_TIMER_H_INCLUDED
