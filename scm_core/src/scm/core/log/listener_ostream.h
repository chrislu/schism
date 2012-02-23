
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LISTENER_OSTREAM_H_INCLUDED
#define SCM_CORE_LOG_LISTENER_OSTREAM_H_INCLUDED

#include <ostream>

#include <scm/core/log/listener.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class message;

class __scm_export(core) listener_ostream : public listener
{
public:
    listener_ostream(std::ostream& os);
    virtual ~listener_ostream();

    void                notify(const message& msg);

private:
    std::ostream&       _ostream;

}; // class listener_ostream

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_LISTENER_OSTREAM_H_INCLUDED
