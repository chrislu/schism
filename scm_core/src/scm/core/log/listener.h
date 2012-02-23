
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LISTENER_H_INCLUDED
#define SCM_CORE_LOG_LISTENER_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class message;

class __scm_export(core) listener
{
public:
    typedef enum {
        log_plain,          // raw message
        log_decorated,      // logger <level>: message
        log_full_decorated  // date time: logger: message
    } log_style;
public:
    listener();
    virtual ~listener();

    virtual void            notify(const message& msg) = 0;
    log_style               style() const;
    void                    style(log_style s);

protected:
    std::string             get_log_message(const message& msg);

private:
    log_style               _style;


}; // class listener

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_LISTENER_H_INCLUDED
