
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "log.h"

#include <scm/core/log/core.h>
#include <scm/core/platform/platform.h>

namespace  {

scm::log::logger& default_out = scm::logger("scm");

} // namespace 

namespace scm {

log::logger&
logger(const std::string& name)
{
    log::logger& ret_log = log::core::get().get_logger(name);
    return (ret_log);
}

log::out_stream
out()
{
    //log::logger& ret_log = log::core::get().get_logger("scm.out");
    //return (ret_log.output());
    return (default_out.output());
}

log::out_stream
err()
{
    //log::logger& ret_log = log::core::get().get_logger("scm.err");
    //return (ret_log.error());
#if SCM_DEBUG
    return (default_out.debug());
#else
    return (default_out.error());
#endif
}

} // namespace scm
