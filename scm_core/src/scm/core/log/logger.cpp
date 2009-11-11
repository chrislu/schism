
#include "logger.h"

#include <cassert>

#include <scm/core/log/listener.h>
#include <scm/core/log/message.h>
#include <scm/core/log/out_stream.h>
#include <scm/core/utilities/foreach.h>

namespace scm {
namespace logging {

logger::logger(const std::string&       log_name,
               level_type               log_lev,
               scm::shared_ptr<logger>  parent)
  : _name(log_name),
    _log_level(log_lev),
    _parent(parent)
{
}

logger::~logger()
{
    clear_listeners();
}

const level&
logger::log_level() const
{
    return (_log_level);
}

void
logger::log_level(level_type lev)
{
    _log_level = lev;
}

const std::string&
logger::name() const
{
    return (_name);
}

void
logger::log(const level& lev, const std::string& msg)
{
    process_message(message(*this, lev, msg));
}

out_stream
logger::trace()
{
    return (out_stream(ll_trace, *this));
}

out_stream
logger::debug()
{
    return (out_stream(ll_debug, *this));
}

out_stream
logger::output()
{
    return (out_stream(ll_output, *this));
}

out_stream
logger::info()
{
    return (out_stream(ll_info, *this));
}

out_stream
logger::warn()
{
    return (out_stream(ll_warning, *this));
}

out_stream
logger::error()
{
    return (out_stream(ll_error, *this));
}

out_stream
logger::fatal()
{
    return (out_stream(ll_fatal, *this));
}

void
logger::process_message(const message& msg)
{
    foreach (const listener_ptr& listn_ptr, _listeners) {
        listn_ptr->notify(msg);
    }
    if (_parent) {
        _parent->process_message(msg);
    }
}

void
logger::add_listener(const listener_ptr l)
{
    assert(l);

    boost::mutex::scoped_lock lock(_listeners_mutex);
    _listeners.insert(l);
}

void
logger::del_listener(const listener_ptr l)
{
    assert(l);

    boost::mutex::scoped_lock lock(_listeners_mutex);
    _listeners.erase(l);
}

void
logger::clear_listeners()
{
    boost::mutex::scoped_lock lock(_listeners_mutex);
    _listeners.clear();
}

} // namespace logging
} // namespace scm
