
#ifndef SCM_CORE_LOG_OUT_STREAM_H_INCLUDED
#define SCM_CORE_LOG_OUT_STREAM_H_INCLUDED

#include <string>
#include <sstream>

#include <boost/format/format_fwd.hpp>

#include <scm/core/log/level.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace logging {

class logger;

class __scm_export(core) out_stream
{
public:
    out_stream(scm::logging::level_type log_lev,
               scm::logging::logger&    ref_logger);
    out_stream(const out_stream& os);
    virtual ~out_stream();

    out_stream&             operator=(const out_stream& os);

    const level&            log_level() const;
    void                    flush();

private:
    logger*                 _logger;
    level                   _log_level;

    std::ostringstream      _ostream;
public:
    template <typename T> out_stream& operator<<(const T& rhs);
    out_stream& operator<<(const scm::logging::level& lev);
    //friend out_stream& operator<<(out_stream& os, const boost::format& rhs);
    out_stream& operator<<(std::ios_base& (*_Pfn)(std::ios_base&));
    out_stream& operator<<(std::ostream& (*_Pfn)(std::ostream&));

}; // out_stream

} // namespace logging
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#include "out_stream.inl"

#endif // SCM_CORE_LOG_OUT_STREAM_H_INCLUDED
