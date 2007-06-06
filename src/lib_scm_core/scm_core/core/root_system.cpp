
#include "root_system.h"

#include <cassert>

#include <boost/lexical_cast.hpp>

#include <scm_core/core.h>

#include <scm_core/console/console_system.h>
#include <scm_core/console/console_output_listener_stdout.h>
#include <scm_core/core/root_system.h>
#include <scm_core/core/core_system_singleton.h>
#include <scm_core/core/version.h>
#include <scm_core/exception/system_exception.h>
#include <scm_core/resource/resource_manager.h>
#include <scm_core/time/time_system.h>

namespace scm {

// initialize extern singleton instances
core::core_system_singleton<core::root_system>::type        root        = core::core_system_singleton<core::root_system>::type();
core::core_system_singleton<con::console_system>::type      console     = core::core_system_singleton<con::console_system>::type();
core::core_system_singleton<time::time_system>::type        timing      = core::core_system_singleton<time::time_system>::type();

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
    console.set_instance(_console.get());
    register_subsystem("console", _console.get());

    _console_listener.reset(new scm::con::console_output_listener_stdout(*_console.get()));

    console.get() << con::log_level(con::output)
                  << get_version_string()
                  << std::endl;

    // timing
    _timing.reset(new time::time_system());
    register_subsystem("timing", _timing.get());

    console.get() << con::log_level(con::info)
                  << "initializing subsystems:"  << std::endl;
    if (!initialize_subsystems()) {
        console.get() << con::log_level(con::error)
                      << "root_system::initialize(): "
                      << "unable to initialize subsystems" << std::endl;
        return (false);

    }
    // set global singleton access
    setup_global_system_singletons();

    console.get() << con::log_level(con::info)
                  << "initialization successful"  << std::endl;
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

std::string root_system::get_version_string() const
{
    using boost::lexical_cast;

    std::string output;

    output = SCHISM_NAME + std::string(" (") +
             lexical_cast<std::string>(VERSION_MAJOR) + std::string(".") +
             lexical_cast<std::string>(VERSION_MINOR) + std::string(".") +
             lexical_cast<std::string>(VERSION_REVISION) + std::string("_") +
             VERSION_TAG + std::string("_") +
             VERSION_BUILD_TAG + std::string(") '") +
             VERSION_NAME  + std::string("'");

    return (output);

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
    scm::timing.set_instance(_timing.get());
}

void root_system::reset_global_system_singletons()
{
    scm::root.set_instance(0);
    scm::console.set_instance(0);
    scm::timing.set_instance(0);
}

// subsystems
scm::core::system& root_system::get_subsystem(const std::string& sys_name)
{
    system_container::size_type         index;
    system_assoc_container::iterator    prev = _subsystems_named.find(sys_name);

    if (prev != _subsystems_named.end()) {
        index = prev->second;
        
        assert(_subsystems[index].first != 0);
        
        return *(_subsystems[index].first);
    }
    else {
        std::stringstream output;

        output << "root_system::get_subsystem(): "
               << "subsystem '" << sys_name << "' not found" << std::endl;

        console.get() << con::log_level(con::error)
                      << output.str();

        throw scm::core::system_exception(output.str());
    }
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
        console.get() << con::log_level(con::info)
                      << " - initializing: '" << sys_it->second << "'" << std::endl;

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
        console.get() << con::log_level(con::info)
                      << " - shutting down: '" << sys_it->second << "'" << std::endl;
        if (!sys_it->first->shutdown()) {
            console.get() << con::log_level(con::error)
                          << "root_system::shutdown_subsystems(): "
                          << "subsystem '" << sys_it->second << "' shutdown returned with error" << std::endl;
        }
    }
}

// resource managers
void root_system::register_resource_manager(scm::res::resource_manager_base*const man,
                                            std::size_t                           id)
{
    resource_manager_container::iterator man_it = _resource_managers.find(id);
    
    if (man_it != _resource_managers.end()) {
        console.get() << con::log_level(con::warning)
                      << "root_system::register_resource_manager(): "
                      << "resource manager (id = '" << id << "') allready registered: "
                      << "overriding present instance" << std::endl;
        
        man_it->second->shutdown();

        _resource_managers.erase(man_it);
    }

    _resource_managers.insert(resource_manager_container::value_type(id, man));
}

void root_system::unregister_resource_manager(std::size_t id)
{
    resource_manager_container::iterator man_it = _resource_managers.find(id);
    
    if (man_it != _resource_managers.end()) {
        
        assert(man_it->second != 0);
        
        man_it->second->shutdown();
        _resource_managers.erase(man_it);
    }
}

scm::res::resource_manager_base& root_system::get_resource_manager(std::size_t id)
{
    resource_manager_container::iterator man_it = _resource_managers.find(id);
    
    if (man_it != _resource_managers.end()) {
        
        assert(man_it->second != 0);
        
        return (*(man_it->second));
    }
    else {
        std::stringstream output;

        output << "root_system::get_resource_manager(): "
               << "resource manager (id = '" << id << "') not found" << std::endl;

        console.get() << con::log_level(con::error)
                      << output.str();

        throw scm::core::system_exception(output.str());
    }
}

void root_system::unregister_resource_managers()
{
    resource_manager_container::iterator man_it;

    for (man_it  = _resource_managers.begin();
         man_it != _resource_managers.end();
         ++man_it) {

        assert(man_it->second != 0);
        man_it->second->shutdown();
    }

    _resource_managers.clear();
}
