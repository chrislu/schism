
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "core.h"

#include <iostream>
#include <cassert>

#include <scm/core/log/logger.h>
#include <scm/core/utilities/foreach.h>

SCM_SINGLETON_PLACEMENT(core, scm::log::logging_core)

namespace scm {
namespace log {

logging_core::logging_core()
: default_log_name("")
{
    // create default logger
    _loggers[default_log_name].reset(new logger(default_log_name, ll_output, logger_ptr()));
    _default_logger = _loggers[default_log_name];

    assert(!_default_logger.expired());
}

logging_core::~logging_core()
{
    foreach_reverse (logger_container::value_type& log_it, _loggers) {
        if (!log_it.second.unique()) {
            std::cerr << "logging_core::~logging_core(): <error> possible dangeling logger instance ("
                      << log_it.second->name()
                      << ")" << std::endl;
        }
        log_it.second.reset();
    }
}

logger&
logging_core::default_log() const
{
    assert(!_default_logger.expired());

    return (*(_default_logger.lock()));
}

logger&
logging_core::get_logger(const std::string& log_name)
{
    return (*get_logger_ptr(log_name));
}

logging_core::logger_ptr
logging_core::get_logger_ptr(const std::string& log_name)
{
    if (log_name == default_log_name) {
        return (_default_logger.lock());
    }
    else {
        //bool found_logger = false;
        { // mutex lock scope
            boost::mutex::scoped_lock       lock(_loggers_mutex);
            logger_container::iterator      log_iter = _loggers.find(log_name);

            if (log_iter != _loggers.end()) {
                //found_logger = true;
                return (log_iter->second);
            }
        }

        //*if (!found_logger)*/
        { // mutex lock scope
            logger_ptr parent_log = get_logger_ptr(retrieve_parent_name(log_name));

            // ok this logger does not exist yet
            logger_ptr new_log(new logger(log_name, ll_output, parent_log));
            _loggers[log_name] = new_log;
            return (new_log);
        }
    }
}

std::string
logging_core::retrieve_parent_name(const std::string& name) const
{
    if (name.find("..") != std::string::npos) {
        throw std::invalid_argument("logging_core::retrieve_parent_name: <fatal> encountered logger name containing '..'");
    }
    else if (name.find(".") == name.size() - 1) {
        throw std::invalid_argument("logging_core::retrieve_parent_name: <fatal> encountered logger name ending with '.'");
    }

    std::string::size_type parent_end_index = name.rfind(".");
    if (parent_end_index != std::string::npos) {
        return (name.substr(0, parent_end_index));
    }
    else {
        return (default_log_name);
    }
}

std::ostream& operator<<(std::ostream& os, const logging_core& rhs)
{
    os << "logging_core registered loggers:" << std::endl;

    foreach_reverse (const logging_core::logger_container::value_type& log_it, rhs._loggers) {
        os << " - '" << log_it.second->name() << "' with ptr instances: " << log_it.second.use_count() << std::endl;
    }

    return (os);
}

} // namespace log
} // namespace scm
