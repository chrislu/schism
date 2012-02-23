
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LEVEL_H_INCLUDED
#define SCM_CORE_LOG_LEVEL_H_INCLUDED

#include <string>

#include <boost/operators.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

typedef enum {
    ll_fatal       = 0x01,
    ll_error,
    ll_warning,
    ll_info,
    ll_output,
    ll_debug,
    ll_trace
} level_type;

class __scm_export(core) level : boost::less_than_comparable<level,
                                 boost::less_than_comparable<level, level_type, 
                                 boost::equality_comparable<level,
                                 boost::equality_comparable<level, level_type> > > >
{
public:
    level(level_type lev);
    level(const level& lev);

    level&              operator=(const level& rhs);
    level&              operator=(const level_type& rhs);

    bool                operator==(const level& rhs) const;
    bool                operator==(const level_type& rhs) const;

    bool                operator<(const level& rhs) const;
    bool                operator<(const level_type& rhs) const;

    level_type          log_level() const;
    const std::string&  to_string() const;

private:
    level_type          _log_level;

}; // class level

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_LEVEL_H_INCLUDED
