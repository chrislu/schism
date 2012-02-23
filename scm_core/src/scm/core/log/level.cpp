
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "level.h"

namespace scm {
namespace log {

level::level(level_type lev)
  : _log_level(lev)
{
}

level::level(const level& lev)
  : _log_level(lev._log_level)
{
}

level_type
level::log_level() const
{
    return (_log_level);
}

const std::string&
level::to_string() const
{
    static std::string  level_strings[] = {"fatal", "error", "warning", "info", "output", "debug", "trace"};

    return (level_strings[_log_level - ll_fatal]);
}

level&
level::operator=(const level& rhs)
{
    _log_level = rhs.log_level();

    return (*this);
}

level&
level::operator=(const level_type& rhs)
{
    _log_level = rhs;

    return (*this);
}

bool
level::operator==(const level& rhs) const
{
    return (_log_level == rhs.log_level());
}

bool
level::operator==(const level_type& rhs) const
{
    return (_log_level == rhs);
}

bool
level::operator<(const level& rhs) const
{
    return (_log_level < rhs.log_level());
}

bool
level::operator<(const level_type& rhs) const
{
    return (_log_level < rhs);
}

} // namespace log
} // namespace scm
