
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LOG_LOGGER_H_INCLUDED
#define SCM_LOG_LOGGER_H_INCLUDED

#include <set>
#include <string>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/memory.h>
#include <scm/core/log/level.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class listener;
class message;
class out_stream;

class __scm_export(core) logger : boost::noncopyable
{
public:
    typedef char                                char_type;
    typedef std::basic_string<char_type>        string_type;
    typedef std::basic_stringstream<char_type>  stream_type;
    typedef std::basic_istringstream<char_type> istream_type;
    typedef std::basic_ostringstream<char_type> ostream_type;

    typedef scm::shared_ptr<logger>             logger_ptr;
    typedef scm::shared_ptr<listener>           listener_ptr;

private:
    typedef std::set<listener_ptr>              listener_container;

public:
    logger(const string_type&       log_name,
           level_type               log_lev,
           scm::shared_ptr<logger>  parent);
    virtual ~logger();

    const level&                    log_level() const;
    void                            log_level(level_type lev);

    const string_type&              name() const;

    void                            log(const level& lev, const string_type& msg);

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

    char_type                       indent_fill_char() const;
    void                            indent_fill_char(char_type c);
    int                             indent_level() const;
    void                            indent_level(int l);
    void                            increase_indent_level();
    void                            decrease_indent_level();
    int                             indent_width() const;
    void                            indent_width(int w);

private:
    void                            process_message(const message& msg);

private:
    logger_ptr                      _parent;
    level                           _log_level;
    string_type                     _name;

    listener_container              _listeners;
    boost::mutex                    _listeners_mutex;

    char_type                       _indent_fill_char;
    int                             _indent_level;
    int                             _max_indent_level;
    int                             _indent_width;

}; // class logger

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_LOG_LOGGER_H_INCLUDED
