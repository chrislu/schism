
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_OUT_STREAM_H_INCLUDED
#define SCM_CORE_LOG_OUT_STREAM_H_INCLUDED

#include <string>
#include <sstream>

#include <boost/format/format_fwd.hpp>

#include <scm/core/log/level.h>
#include <scm/core/log/logger.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class __scm_export(core) out_stream
{
public:
    typedef logger                      logger_type;
    typedef logger_type::char_type      char_type;
    typedef logger_type::string_type    string_type;
    typedef logger_type::ostream_type   ostream_type;

public:
    out_stream(scm::log::level_type log_lev,
               scm::log::logger&    ref_logger);
    out_stream(const out_stream& os);
    virtual ~out_stream();

    out_stream&             operator=(const out_stream& os);

    const level&            log_level() const;
    void                    switch_log_level(const scm::log::level& lev);

    logger&                 associated_logger();
    const logger&           associated_logger() const;

    ostream_type&           ostream();
    const ostream_type&     ostream() const;

    void                    flush();

    template <typename T>
    out_stream&             operator<<(const T& rhs);
    out_stream&             operator<<(out_stream& (*manip_func)(out_stream&));
    out_stream&             operator<<(std::ios_base& (*_Pfn)(std::ios_base&));

protected:
    logger*                 _logger;
    level                   _log_level;
    level                   _message_level;

    ostream_type            _ostream;

}; // out_stream

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#include "out_stream.inl"

#endif // SCM_CORE_LOG_OUT_STREAM_H_INCLUDED
