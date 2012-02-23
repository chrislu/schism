
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_INIT_FUNCTIONS_H_INCLUDED
#define SCM_CORE_INIT_FUNCTIONS_H_INCLUDED

#include <boost/function.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {

class core;

namespace module {

class __scm_export(core) initializer
{
public:
    typedef boost::function<bool (core&)>       initialize_function;
    typedef boost::function<bool (core&)>       shutdown_function;

public:
    static void            add_pre_core_init_function(const initialize_function func);
    static void            add_post_core_init_function(const initialize_function func);

    static void            add_pre_core_shutdown_function(const shutdown_function func);
    static void            add_post_core_shutdown_function(const shutdown_function func);

private:
    static bool            run_pre_core_init_functions(core& c);
    static bool            run_post_core_init_functions(core& c);

    static bool            run_pre_core_shutdown_functions(core& c);
    static bool            run_post_core_shutdown_functions(core& c);

    friend class ::scm::core;
};

// helper structs
struct __scm_export(core) static_initializer
{
    typedef boost::function<void (void)>    static_init_func;

    static_initializer(const static_init_func func);
};

} // namespace module
} // namespace scm

#include "initializer.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_INIT_FUNCTIONS_H_INCLUDED
