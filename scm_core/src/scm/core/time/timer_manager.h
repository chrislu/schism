
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_TIME_TIMER_MANAGER_H_INCLUDED
#define SCM_CORE_TIME_TIMER_MANAGER_H_INCLUDED

#include <scm/core/time/timer_interface.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace time {

class __scm_export(core) timer_manager
{
public:
    typedef boost::shared_ptr<scm::time::timer_interface>   timer_ptr;

public:

    timer_ptr           add_timer(const std::string& name);
    timer_ptr           get_timer(const std::string& name);
    timer_ptr           del_timer(const std::string& name);

protected:

private:

}; // class timer_manager

} // namespace time
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_TIME_TIMER_MANAGER_H_INCLUDED
