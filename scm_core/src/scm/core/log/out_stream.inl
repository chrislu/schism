
#include <boost/format.hpp>

#include <scm/core/log/logger.h>

namespace scm {
namespace logging {

template<typename T>
out_stream&
out_stream::operator<<(const T& rhs)
{
    if (log_level() <= _logger->log_level()) {
        _ostream << rhs;
    }

    return (*this);
}


inline out_stream&
out_stream::operator<<(const level& lev)
{
    if (log_level() != lev) {
        //os.flush();
        _log_level = lev;
    }

    return (*this);
}

#if 0
inline out_stream&
operator<<(out_stream& os, const boost::format& rhs)
{
    if (os.log_level() <= os._logger->log_level()) {
        os._ostream << rhs;
    }

    return (os);
}
#endif
inline out_stream&
out_stream::operator<<(std::ios_base& (*_Pfn)(std::ios_base&))
{
    if (log_level() <= _logger->log_level()) {
        _ostream << _Pfn;
    }

    return (*this);
}

inline out_stream&
out_stream::operator<<(std::ostream& (*_Pfn)(std::ostream&))
{
    if (log_level() <= _logger->log_level()) {
        _ostream << _Pfn;
    }

    return (*this);
}

} // namespace logging
} // namespace scm
