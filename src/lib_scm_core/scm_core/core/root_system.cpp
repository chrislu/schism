
#include "root_system.h"

#include <cassert>

#include <scm_core/core.h>

#include <scm_core/console/console_system.h>
#include <scm_core/core/root_system.h>
#include <scm_core/core/core_system_singleton.h>

#include <scm_core/console/console_output_listener_stdout.h>

namespace scm {

// initialize extern singleton instances
core::core_system_singleton<core::root_system>::type        root        = core::core_system_singleton<core::root_system>::type();
core::core_system_singleton<con::console_system>::type      console     = core::core_system_singleton<con::console_system>::type();

} // namespace scm

using namespace scm::core;

root_system::root_system()
{
}

root_system::~root_system()
{
}

bool root_system::initialize()
{
    if (_initialized) {
        console.get() << con::log_level(con::warning)
                      << "root_system::initialize(): "
                      << "allready initialized" << std::endl;
        return (true);
    }

    // console
    _console.reset(new con::console_system());
    register_subsystem("console", _console.get());

    _console_listener.reset(new scm::con::console_output_listener_stdout(*_console.get()));

    if (!initialize_subsystems()) {
        console.get() << con::log_level(con::error)
                      << "root_system::initialize(): "
                      << "unable to initialize subsystems" << std::endl;
        return (false);

    }
    // set global singleton access
    setup_global_system_singletons();

    _initialized = true;
    return (true);
}

bool root_system::shutdown()
{
    // reset global singleton access
    reset_global_system_singletons();

    shutdown_subsystems();

    _initialized = false;
    return (true);
}

scm::con::console_output_listener& root_system::get_std_console_listener()
{
    assert(_console_listener.get() != 0);

    return (*_console_listener);
}

void root_system::setup_global_system_singletons()
{
    scm::root.set_instance(this);
    scm::console.set_instance(_console.get());
}

void root_system::reset_global_system_singletons()
{
    scm::root.set_instance(0);
    scm::console.set_instance(0);
}

void root_system::register_subsystem(const std::string& sys_name, scm::core::system* sys_ptr)
{
    system_container::size_type         index;
    system_assoc_container::iterator    prev = _subsystems_named.find(sys_name);

    if (prev != _subsystems_named.end()) {
        console.get() << con::log_level(con::warning)
                      << "root_system::register_subsystem(): "
                      << "subsystem '" << sys_name << "' allready registered: "
                      << "overriding present instance" << std::endl;
        index = prev->second;
        _subsystems[index].first = sys_ptr;
    }
    else {
        _subsystems.push_back(std::make_pair(sys_ptr, sys_name));
        index = _subsystems.size() - 1;
        _subsystems_named.insert(system_assoc_container::value_type(sys_name, index));
    }
}

void root_system::unregister_subsystem(const std::string& sys_name)
{
    system_container::size_type         index;
    system_assoc_container::iterator    prev = _subsystems_named.find(sys_name);

    if (prev != _subsystems_named.end()) {
        index = prev->second;
        _subsystems.erase(_subsystems.begin() + index);
        _subsystems_named.erase(prev);
    }
    else {
        console.get() << con::log_level(con::warning)
                      << "root_system::unregister_subsystem(): "
                      << "subsystem '" << sys_name << "' not registered" << std::endl;
    }
}

void root_system::unregister_subsystems()
{
    _subsystems.clear();
    _subsystems_named.clear();
}

bool root_system::initialize_subsystems()
{
    system_container::iterator      sys_it;

    for (sys_it = _subsystems.begin(); sys_it != _subsystems.end(); ++sys_it) {
        if (!sys_it->first->initialize()) {
            console.get() << con::log_level(con::error)
                          << "root_system::initialize_subsystems(): "
                          << "subsystem '" << sys_it->second << "' initialize returned with error" << std::endl;
            return (false);
        }
    }

    return (true);
}

void root_system::shutdown_subsystems()
{
    system_container::reverse_iterator    sys_it;

    for (sys_it = _subsystems.rbegin(); sys_it != _subsystems.rend(); ++sys_it) {
        if (!sys_it->first->shutdown()) {
            console.get() << con::log_level(con::error)
                          << "root_system::shutdown_subsystems(): "
                          << "subsystem '" << sys_it->second << "' shutdown returned with error" << std::endl;
        }
    }
}
