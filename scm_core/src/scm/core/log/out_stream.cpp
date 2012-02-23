
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "out_stream.h"

#include <boost/utility.hpp>

#include <scm/core/log/logger.h>

namespace scm {
namespace log {

out_stream::out_stream(scm::log::level_type log_lev,
                       scm::log::logger&    ref_logger)
  : _log_level(log_lev),
    _message_level(log_lev),
    _logger(boost::addressof(ref_logger))
{
}

out_stream::out_stream(const out_stream& os)
  : _log_level(os._log_level),
    _message_level(os._message_level),
    _logger(os._logger)
{
}

out_stream::~out_stream()
{
    flush();
}

out_stream&
out_stream::operator=(const out_stream& os)
{
    _log_level      = os._log_level;
    _message_level  = os._message_level;
    _logger         = os._logger;

    return (*this);
}

const level&
out_stream::log_level() const
{
    return (_log_level);
}

void
out_stream::switch_log_level(const scm::log::level& lev)
{
    if (_message_level != lev) {
        flush();
    }
    _message_level = lev;
}

logger&
out_stream::associated_logger()
{
    return (*_logger);
}

const logger&
out_stream::associated_logger() const
{
    return (*_logger);
}

void
out_stream::flush()
{
    if (!_ostream.str().empty()) {
        _logger->log(_message_level, _ostream.str());
    }

    _ostream.clear();
    _ostream.str("");
}

out_stream::ostream_type&
out_stream::ostream()
{
    return (_ostream);
}

const out_stream::ostream_type&
out_stream::ostream() const
{
    return (_ostream);
}

out_stream&
out_stream::operator<<(out_stream& (*manip_func)(out_stream&))
{
    if (_message_level <= _log_level) {
        (*manip_func)(*this);
    }
    return (*this);
}

out_stream&
out_stream::operator<<(std::ios_base& (*_Pfn)(std::ios_base&))
{
    if (_message_level <= _log_level) {
        _ostream << _Pfn;
    }
    return (*this);
}

} // namespace log
} // namespace scm
