
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "logger.h"

#include <cassert>

#include <scm/core/log/listener.h>
#include <scm/core/log/message.h>
#include <scm/core/log/out_stream.h>
#include <scm/core/utilities/foreach.h>

namespace scm {
namespace log {

logger::logger(const string_type&       log_name,
               level_type               log_lev,
               scm::shared_ptr<logger>  parent)
  : _name(log_name),
    _log_level(log_lev),
    _parent(parent),
    _indent_fill_char(char_type(' ')),
    _indent_level(0),
    _max_indent_level(8),
    _indent_width(4)
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

const logger::string_type&
logger::name() const
{
    return (_name);
}

void
logger::log(const level& lev, const string_type& msg)
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
    if (msg.log_level() <= _log_level) {
        boost::mutex::scoped_lock lock(_listeners_mutex);
        foreach (const listener_ptr& listn_ptr, _listeners) {
            listn_ptr->notify(msg);
        }
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

logger::char_type
logger::indent_fill_char() const
{
    return (_indent_fill_char);
}

int
logger::indent_level() const
{
    return (_indent_level);
}

int
logger::indent_width() const
{
    return (_indent_width);
}

void
logger::indent_fill_char(char_type c)
{
    _indent_fill_char = c;
}

void
logger::indent_level(int l)
{
    _indent_level = l;
}

void
logger::increase_indent_level()
{
    _indent_level = (_indent_level >= _max_indent_level) ? _max_indent_level : _indent_level + 1;
}

void
logger::decrease_indent_level()
{
    _indent_level = (_indent_level <= 0) ? 0 : _indent_level - 1;
}

void
logger::indent_width(int w)
{
    _indent_width = w;
}

} // namespace log
} // namespace scm
