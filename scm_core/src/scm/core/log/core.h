
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_CORE_H_INCLUDED
#define SCM_CORE_LOG_CORE_H_INCLUDED

#include <map>
#include <string>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/utility.hpp>
#include <boost/thread/mutex.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/memory.h>
#include <scm/core/utilities/singleton.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class logger;

class __scm_export(core) logging_core : boost::noncopyable
{
public:
    const std::string                           default_log_name;

private:
    typedef scm::shared_ptr<logger>             logger_ptr;
    typedef std::map<std::string, logger_ptr>   logger_container;

public:
    logging_core();
    virtual ~logging_core();

    logger&                                     default_log() const;
    logger&                                     get_logger(const std::string& log_name);

private:
    std::string                                 retrieve_parent_name(const std::string& name) const;
    logger_ptr                                  get_logger_ptr(const std::string& log_name);

private:
    logger_container                            _loggers;
    boost::mutex                                _loggers_mutex;

    scm::weak_ptr<logger>                       _default_logger;

    friend __scm_export(core) std::ostream& operator<<(std::ostream& os, const logging_core& rhs);

}; // class core

typedef singleton<logging_core>    core;

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_CORE_H_INCLUDED
