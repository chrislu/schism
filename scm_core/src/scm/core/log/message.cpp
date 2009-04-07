
#include "message.h"

#include <sstream>

#include <scm/time.h>
#include <scm/core/log/logger.h>

namespace scm {
namespace logging {

message::message(const logger& ref_log, const level& lev, const std::string& msg)
  : _sending_logger(ref_log),
    _log_level(lev),
    _message(msg)
{
    _date   = time::universal_date();
    _time   = time::universal_time();
}

message::~message()
{
}


const logger&
message::sending_logger() const
{
    return (_sending_logger);
}

const level&
message::log_level() const
{
    return (_log_level);
}

const std::string&
message::raw_message() const
{
    return (_message);
}

std::string
message::decorated_message() const
{
    std::ostringstream dec_message;

    if (!sending_logger().name().empty()) {
        dec_message << sending_logger().name() << " ";
    }
    dec_message << "<" << log_level().to_string() << "> ";
    dec_message << raw_message();

    return (dec_message.str());
}

std::string
message::full_decorated_message() const
{
    std::ostringstream dec_message;

    dec_message << _time << ": ";

    if (!sending_logger().name().empty()) {
        dec_message << sending_logger().name() << " ";
    }
    dec_message << "<" << log_level().to_string() << "> ";
    dec_message << raw_message();

    return (dec_message.str());
}

} // namespace logging
} // namespace scm
