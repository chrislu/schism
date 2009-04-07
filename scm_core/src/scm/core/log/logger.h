
#ifndef SCM_LOG_LOGGER_H_INCLUDED
#define SCM_LOG_LOGGER_H_INCLUDED

#include <set>
#include <string>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/thread/mutex.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/pointer_types.h>
#include <scm/core/log/level.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace logging {

class listener;
class message;
class out_stream;

class __scm_export(core) logger
{
public:
    typedef scm::shared_ptr<logger>     logger_ptr;
    typedef scm::shared_ptr<listener>   listener_ptr;

private:
    typedef std::set<listener_ptr>      listener_container;

public:
    logger(const std::string&       log_name,
           level_type               log_lev,
           scm::shared_ptr<logger>  parent);
    virtual ~logger();

    const level&                    log_level() const;
    void                            log_level(level_type lev);

    const std::string&              name() const;

    void                            log(const level& lev, const std::string& msg);

    out_stream                      trace();
    out_stream                      debug();
    out_stream                      output();
    out_stream                      info();
    out_stream                      warn();
    out_stream                      error();
    out_stream                      fatal();

    void                            add_listener(const listener_ptr l);
    void                            del_listener(const listener_ptr l);
    void                            clear_listeners();

private:
    void                            process_message(const message& msg);

private:
    logger_ptr                      _parent;
    level                           _log_level;
    std::string                     _name;

    listener_container              _listeners;
    boost::mutex                    _listeners_mutex;

}; // class logger

} // namespace logging
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_LOG_LOGGER_H_INCLUDED
