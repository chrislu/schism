
#include "log.h"

#include <scm/core/log/log_core.h>

namespace scm {

log&
logger(const std::string& name)
{
    log& ret_log = logging::core::get().get_logger(name);
    return (ret_log);
}

logging::out_stream
out()
{
    log& ret_log = logging::core::get().get_logger("scm.out");
    return (ret_log.output());
}

logging::out_stream
err()
{
    log& ret_log = logging::core::get().get_logger("scm.err");
    return (ret_log.error());
}

} // namespace scm
