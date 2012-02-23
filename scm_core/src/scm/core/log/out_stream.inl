
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <boost/format.hpp>

#include <scm/core/log/logger.h>

namespace scm {
namespace log {

template<typename T>
out_stream&
out_stream::operator<<(const T& rhs)
{
    if (_message_level <= _log_level) {
        _ostream << rhs;
    }

    return (*this);
}

} // namespace log
} // namespace scm
