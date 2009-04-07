
#include "listener_ostream.h"

#include <scm/core/log/message.h>

namespace scm {
namespace logging {

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

} // namespace logging
} // namespace scm
