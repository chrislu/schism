
#ifndef SCM_CORE_LOG_MESSAGE_H_INCLUDED
#define SCM_CORE_LOG_MESSAGE_H_INCLUDED

#include <boost/utility.hpp>

#include <scm/core/log/level.h>
#include <scm/core/time/time_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace logging {

class logger;

class __scm_export(core) message : boost::noncopyable
{
public:
    message(const logger& ref_log, const level& lev, const std::string& msg);
    virtual ~message();

    const logger&           sending_logger() const;
    const level&            log_level() const;
    const std::string&      raw_message() const;
    std::string             decorated_message() const;
    std::string             full_decorated_message() const;

private:
    const logger&           _sending_logger;
    level                   _log_level;
    std::string             _message;

    time::date              _date;
    time::ptime             _time;

    // thread id
    // process id

}; // class listener

} // namespace logging
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_CORE_LOG_MESSAGE_H_INCLUDED
