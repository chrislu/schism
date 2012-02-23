
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LOGGER_STATE_H_INCLUDED
#define SCM_CORE_LOG_LOGGER_STATE_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/log/logger.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class out_stream;

class __scm_export(core) logger_format_saver : boost::noncopyable
{
    typedef logger                      logger_type;
    typedef logger_type::char_type      char_type;

public:
    logger_format_saver(logger_type& l);
    //logger_format_saver(out_stream& los); // todo rvalue ref
    ~logger_format_saver();

    void                restore();

private:
    logger_type&        _logger;

    char_type           _s_indent_fill_char;
    int                 _s_indent_level;
    int                 _s_indent_width;

}; // class logger_format_saver

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_CORE_LOG_LOGGER_STATE_H_INCLUDED
