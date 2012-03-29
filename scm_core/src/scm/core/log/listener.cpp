
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "listener.h"

#include <scm/core/log/message.h>

namespace scm {
namespace log {

listener::listener()
  : _style(log_decorated)
{
}

listener::~listener()
{
}

std::string
listener::get_log_message(const message& msg)
{
    switch (_style) {
        case log_plain:             return (msg.plain_message());break;
        case log_decorated:         return (msg.decorated_message());break;
        case log_full_decorated:    return (msg.full_decorated_message());break;
        default:                    return (std::string(""));break;
    }
}

listener::log_style
listener::style() const
{
    return (_style);
}

void
listener::style(log_style s)
{
    _style = s;
}

} // namespace log
} // namespace scm
