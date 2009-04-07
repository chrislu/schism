
#include <boost/format.hpp>

#include <scm/core/log/logger.h>

namespace scm {
namespace logging {

template<class T>
out_stream&
operator<<(out_stream& os, const T& rhs)
{
    if (os.log_level() <= os._logger->log_level()) {
        os._ostream << rhs;
    }

    return (os);
}

inline out_stream&
operator<<(out_stream& os, const level& lev)
{
    if (os.log_level() != lev) {
        os.flush();
        os._log_level = lev;
    }

    return (os);
}


inline out_stream&
operator<<(out_stream& os, const boost::format& rhs)
{
    if (os.log_level() <= os._logger->log_level()) {
        os._ostream << rhs;
    }

    return (os);
}

inline out_stream&
operator<<(out_stream& os, std::ios_base& (*_Pfn)(std::ios_base&))
{
    if (os.log_level() <= os._logger->log_level()) {
        os._ostream << _Pfn;
    }

    return (os);
}

inline out_stream&
operator<<(out_stream& os, std::ostream& (*_Pfn)(std::ostream&))
{
    if (os.log_level() <= os._logger->log_level()) {
        os._ostream << _Pfn;
    }

    return (os);
}

} // namespace logging
} // namespace scm
