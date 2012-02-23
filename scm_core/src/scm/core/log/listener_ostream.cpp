
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "listener_ostream.h"

#include <scm/core/log/message.h>

namespace scm {
namespace log {

listener_ostream::listener_ostream(std::ostream& os)
  : _ostream(os)
{
}

listener_ostream::~listener_ostream()
{
}

void
listener_ostream::notify(const message& msg)
{
    _ostream << get_log_message(msg);
    _ostream.flush();
}

} // namespace log
} // namespace scm
