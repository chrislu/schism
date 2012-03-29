
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_MESSAGE_H_INCLUDED
#define SCM_CORE_LOG_MESSAGE_H_INCLUDED

#include <boost/utility.hpp>

#include <scm/core/log/level.h>
#include <scm/core/log/logger.h>
#include <scm/core/time/time_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class __scm_export(core) message : boost::noncopyable
{
public:
    typedef logger                      logger_type;
    typedef logger_type::char_type      char_type;
    typedef logger_type::string_type    string_type;
    typedef logger_type::stream_type    stream_type;
    typedef logger_type::istream_type   istream_type;
    typedef logger_type::ostream_type   ostream_type;

public:
    message(const logger_type& ref_log, const level& lev, const string_type& msg);
    virtual ~message();

    const logger_type&      sending_logger() const;
    const level&            log_level() const;
    const string_type&      raw_message() const;
    const string_type&      plain_message() const;
    const string_type&      decorated_message() const;
    const string_type&      full_decorated_message() const;

    const string_type&      postdec_decoration() const;
    const string_type&      postdec_message() const;

private:
    void                    decorate_message(const string_type& decoration,
                                             const string_type& in_message,
                                                   string_type& out_message) const;

private:
    const logger_type&      _sending_logger;
    level                   _log_level;
    string_type             _message;
    mutable string_type     _plain_message;
    mutable string_type     _decorated_message;
    mutable string_type     _full_decorated_message;

    mutable string_type     _postdec_decoration;
    mutable string_type     _postdec_message;

    time::date              _date;
    time::ptime             _time;


    // thread id
    // process id
}; // class message

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_MESSAGE_H_INCLUDED
