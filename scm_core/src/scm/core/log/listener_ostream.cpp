
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "listener_ostream.h"

#include <scm/core/log/message.h>
#include <scm/core/log/console_color.h>

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
    switch (msg.log_level().log_level()) {
    case ll_fatal:
    case ll_error:
        _ostream << util::bg_red << util::fg_yellow;
        break;
    case ll_warning:
        _ostream << util::bg_black << util::fg_red;
        break;
    case ll_info:
        _ostream << util::bg_black << util::fg_green;
        break;
    case ll_output:
        _ostream << util::bg_black << util::fg_dk_cyan;
        break;
    case ll_debug:
    case ll_trace:
        _ostream << util::bg_blue << util::fg_white;
        break;
    }
    // generate the message strings
    get_log_message(msg);
    _ostream << msg.postdec_decoration();
    _ostream << util::reset_color;
    _ostream << msg.postdec_message();

    //_ostream << get_log_message(msg);

    _ostream.flush();
}

} // namespace log
} // namespace scm
