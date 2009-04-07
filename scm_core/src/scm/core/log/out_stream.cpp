
#include "out_stream.h"

#include <scm/core/log/logger.h>

namespace scm {
namespace logging {

out_stream::out_stream(scm::logging::level_type log_lev,
                       scm::logging::logger&    ref_logger)
  : _log_level(log_lev),
    _logger(&ref_logger)
{
}

out_stream::out_stream(const out_stream& os)
  : _log_level(os._log_level),
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
    _log_level  = os._log_level;
    _logger     = os._logger;

    return (*this);
}

const level&
out_stream::log_level() const
{
    return (_log_level);
}

void
out_stream::flush()
{
    if (!_ostream.str().empty()) {
        _logger->log(log_level(), _ostream.str());
    }

    _ostream.clear();
    _ostream.str(""); //.clear();
}

} // namespace logging
} // namespace scm
