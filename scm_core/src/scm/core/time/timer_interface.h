
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TIMER_INTERFACE_H_INCLUDED
#define TIMER_INTERFACE_H_INCLUDED

#include <scm/core/time/time_types.h>

#include <scm/core/platform/platform.h>

#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace time {

class __scm_export(core) timer_interface
{
public:
    typedef enum {
        nano_seconds,
        micro_seconds,
        milli_seconds
    } resolution_type;

    typedef time_duration       duration_type;

public:
    timer_interface(resolution_type res_type);
    virtual ~timer_interface();

    virtual void                start()                 = 0;
    virtual void                stop()                  = 0;
    virtual void                intermediate_stop()     = 0;
    virtual void                collect_result() const  = 0;

    duration_type               get_time() const;
    resolution_type             resolution() const;

protected:
    mutable duration_type       _duration;

private:
    resolution_type     _res_type;

}; // class timer_interface

} // namespace time
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TIMER_INTERFACE_H_INCLUDED
