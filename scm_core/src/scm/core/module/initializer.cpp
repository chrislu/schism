
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "initializer.h"

#include <vector>

//#include <boost/thread.hpp>

#include <scm/core/utilities/foreach.h>
#include <scm/core/utilities/static_global.h>

namespace  {

typedef std::vector<scm::module::initializer::initialize_function>  init_func_vec;
typedef std::vector<scm::module::initializer::shutdown_function>    shutdown_func_vec;

boost::mutex    _function_vector_mutex;

SCM_STATIC_GLOBAL(init_func_vec, pre_core_init_funcs);
SCM_STATIC_GLOBAL(init_func_vec, post_core_init_funcs);

SCM_STATIC_GLOBAL(shutdown_func_vec, pre_core_shutdown_funcs);
SCM_STATIC_GLOBAL(shutdown_func_vec, post_core_shutdown_funcs);

} // namespace 

namespace scm {
namespace module {

void
initializer::add_pre_core_init_function(const initialize_function func)
{
    // somehow test if it is still possible to insert stuff
    // if (core::check_instance()) if core::instance().state() < startup;

    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    init_func_vec&              init_funcs = pre_core_init_funcs();

    init_funcs.push_back(func);
}

void
initializer::add_post_core_init_function(const initialize_function func)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    init_func_vec&              init_funcs = post_core_init_funcs();

    init_funcs.push_back(func);
}

void
initializer::add_pre_core_shutdown_function(const shutdown_function func)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    shutdown_func_vec&          shutdown_funcs = pre_core_shutdown_funcs();

    shutdown_funcs.push_back(func);
}

void
initializer::add_post_core_shutdown_function(const shutdown_function func)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    shutdown_func_vec&          shutdown_funcs = post_core_shutdown_funcs();

    shutdown_funcs.push_back(func);
}

bool
initializer::run_pre_core_init_functions(core& c)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    init_func_vec&              init_funcs = pre_core_init_funcs();
    bool                        ret_val    = true;

    foreach (init_func_vec::value_type& func, init_funcs) {
        ret_val = ret_val && func(c);

        if (!ret_val) {
            return (false);
        }
    }

    return (true);
}

bool
initializer::run_post_core_init_functions(core& c)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    init_func_vec&              init_funcs = post_core_init_funcs();
    bool                        ret_val    = true;

    foreach (init_func_vec::value_type& func, init_funcs) {
        ret_val = ret_val && func(c);

        if (!ret_val) {
            return (false);
        }
    }

    return (true);
}

bool
initializer::run_pre_core_shutdown_functions(core& c)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    shutdown_func_vec&          shutdown_funcs = pre_core_shutdown_funcs();
    bool                        ret_val        = true;

    foreach_reverse (shutdown_func_vec::value_type& func, shutdown_funcs) {
        ret_val = ret_val && func(c);
    }

    if (!ret_val) {
        return (false);
    }
    else {
        return (true);
    }
}

bool
initializer::run_post_core_shutdown_functions(core& c)
{
    boost::mutex::scoped_lock   lock(_function_vector_mutex);
    shutdown_func_vec&          shutdown_funcs = post_core_shutdown_funcs();
    bool                        ret_val        = true;

    foreach_reverse (shutdown_func_vec::value_type& func, shutdown_funcs) {
        ret_val = ret_val && func(c);
    }

    if (!ret_val) {
        return (false);
    }
    else {
        return (true);
    }
}

// helper structs
static_initializer::static_initializer(const static_init_func func)
{
    func();
}

} // namespace module
} // namespace scm
