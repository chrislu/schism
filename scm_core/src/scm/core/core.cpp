
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "core.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <scm/log.h>
#include <scm/time.h>
#include <scm/core/version.h>
#include <scm/core/log/logger_state.h>
#include <scm/core/log/listener_file.h>
#include <scm/core/log/listener_ostream.h>
#include <scm/core/module/initializer.h>

namespace scm {

// static init
core* core::_instance = 0;

core::core(int argc, char **argv)
  : _system_state(ss_undefined)
{
    if (check_instance()) {
        throw std::logic_error("scm::core::core(): <fatal> only a single core instance allowed");
    }
    else {
        _instance = this;
    }

    setup_logging("");

    scm::out() << version_string() << log::end;
    scm::out() << "startup time: "
               << time::universal_time() << log::end;
    {
        //scm::log::out_stream x = scm::out();
        //log::logger_format_saver save_indent(x);
        log::logger_format_saver save_indent(scm::out().associated_logger());
        scm::out() << log::info
                   << "initializing scm.core:" << log::end;

        scm::out() << log::indent;

        if (!initialize(argc, argv)) {
            scm::err() << log::fatal
                       << "core::core(): errors during core initialization" << log::end;
            shutdown();
            throw std::runtime_error("core::core: <fatal> errors during initialization");
        }
    }

    scm::out() << log::info
               << "successfully initialized scm.core:" << log::end;

    scm::out() << scm::log::core::get() << log::end;
}

core::~core()
{
    {
        log::logger_format_saver save_indent(scm::out().associated_logger());
        scm::out() << log::info
                   << "shutting down scm.core:" << log::end;
        scm::out() << log::indent;

        if (!shutdown()) {
            throw std::runtime_error("core::~core: <fatal> errors during shutdown");
        }
    }

    scm::out() << log::info
               << "successfully shut down scm.core" << log::end;
    scm::out() << log::info
               << "shutdown time: "
               << time::universal_time() << log::end;
    scm::out() << "bye sick, sad world..." << log::end;

    cleanup_logging();

    _instance = 0;
}

bool
core::check_instance()
{
    return (_instance != 0);
}

core&
core::instance()
{
    assert(check_instance());

    return (*_instance);
}

core::system_state_type
core::system_state() const
{
    return (_system_state);
}

std::string
core::version_string() const
{
    using boost::lexical_cast;

    std::string output;

    output = SCHISM_NAME + std::string(" (") +
             lexical_cast<std::string>(VERSION_MAJOR) + std::string(".") +
             lexical_cast<std::string>(VERSION_MINOR) + std::string(".") +
             lexical_cast<std::string>(VERSION_REVISION) + std::string("_") +
             VERSION_ARCH_TAG + std::string("_") +
             VERSION_TAG + std::string("_") +
             VERSION_BUILD_TAG + std::string(") '") +
             VERSION_NAME  + std::string("'");

    return (output);
}

const core::command_line_option_desc&
core::command_line_options() const
{
    return (_command_line_options);
}

void
core::add_command_line_options(const core::command_line_option_desc&   opt,
                               const std::string&                      module)
{
    command_line_desc_container::iterator mod = _module_options.find(module);
    if (mod != _module_options.end()) {
        // the options for this module are allready registered
        mod->second.add(opt);
    }
    else {
        _module_options.insert(command_line_desc_container::value_type(module, opt));
    }

    _command_line_options.add(opt);
}

core::command_line_position_desc&
core::command_line_positions()
{
    return (_command_line_positions);
}

bool
core::initialize(int argc, char **argv)
{
    scm::out() << log::info
               << " - running pre core init functions" << log::end;

    _system_state = ss_pre_init;
    if (!module::initializer::run_pre_core_init_functions(instance())) {
        scm::err() << log::fatal
                   << "core::initialize(): errors in pre core init functions" << log::end;
        return (false);
    }

    // ok initialize the core
    _system_state = ss_init;

    _command_line_options.add_options()
            ("help", "show this help message");

    scm::out() << log::info
               << " - parsing command line options" << log::end;
    if (!parse_command_line(argc, argv)) {
        scm::err() << log::fatal
                   << "core::initialize(): error parsing command line options" << log::end;
        return (false);
    }

    scm::out() << log::info
               << " - running post core init functions" << log::end;

    if (!module::initializer::run_post_core_init_functions(instance())) {
        scm::err() << log::fatal
                   << "core::initialize(): errors in post core init functions" << log::end;
        return (false);
    }

    _system_state = ss_running;

    return (true);
}

bool
core::shutdown()
{
    _system_state = ss_shutdown;

    scm::out() << log::info
               << " - running pre core shutdown functions" << log::end;

    if (!module::initializer::run_pre_core_shutdown_functions(instance())) {
        scm::err() << log::fatal
                   << "core::shutdown(): errors in pre core shutdown functions" << log::end;
        return (false);
    }

    // shutdown core

    scm::out() << log::info
               << " - running post core shutdown functions" << log::end;

    if (!module::initializer::run_post_core_shutdown_functions(instance())) {
        scm::err() << log::fatal
                   << "core::shutdown(): errors in post core shutdown functions" << log::end;
        return (false);
    }

    return (true);
}

void
core::setup_logging(const std::string& log_file)
{
    if (!log_file.empty()) {
        try {
            log::logger::listener_ptr   logfile_list(new log::listener_file(log_file));
            logfile_list->style(scm::log::listener::log_full_decorated);
            log::core::get().default_log().add_listener(logfile_list);
        }
        catch (std::exception& e) {
            std::cerr << "core::setup_logging(): <error> " << e.what() << std::endl;
        }
#if SCM_DEBUG
        log::core::get().default_log().log_level(log::ll_info);
#else
        log::core::get().default_log().log_level(log::ll_error);
#endif
    }

    // the default output logger
    log::logger::listener_ptr   cout_list(new log::listener_ostream(std::cout));
    cout_list->style(scm::log::listener::log_decorated);//log_full_decorated);//
#if SCM_DEBUG
    logger("scm").log_level(log::ll_debug);
#else
    logger("scm").log_level(log::ll_output);
#endif
    logger("scm").add_listener(cout_list);
}

void
core::cleanup_logging()
{
    logger("scm").clear_listeners();
    log::core::get().default_log().clear_listeners();
}

bool
core::parse_command_line(int argc, char **argv)
{
    bool continue_after_parse = true;

    try {
        namespace bpo = boost::program_options;
        
        //bpo::parsed_options      parsed_cmd_line =  bpo::parse_command_line(argc, argv, _command_line_options);

        //bpo::store(parsed_cmd_line, _command_line);
        //bpo::notify(_command_line);

        bpo::parsed_options parsed_cmd_line = bpo::command_line_parser(argc, argv)
                                                   .options(_command_line_options)
                                                   .positional(_command_line_positions)
                                                   .allow_unregistered()
                                                   .run();
        bpo::store(parsed_cmd_line, _command_line);
        bpo::notify(_command_line);
    }
    catch (std::exception& e) {
        scm::err() << log::fatal
                   << "core::parse_command_line(): error parsing command line (" << e.what() << ")" << log::end;
        // print out usage
        scm::err() << log::fatal
                   << "usage: " << log::nline
                   << _command_line_options;

        continue_after_parse = false;
    }

    if (continue_after_parse) {
        using std::cout;
        using std::string;

        if (_command_line.count("help")) {
            cout << "usage: " << std::endl;
            cout << _command_line_options;

            continue_after_parse = false;
        }
        if (_command_line.count("help-module")) {
            const string& mod = _command_line["help-module"].as<string>();

            command_line_desc_container::iterator mod_iter = _module_options.find(mod);
            if (mod_iter != _module_options.end()) {
                cout << "module '" << mod << "' usage: " << log::end;
                cout << mod_iter->second << log::end;
            }
            else {
                cout << "unknown module '" << mod << "' in the --help-module option" << log::end;
            }
            continue_after_parse = false;
        }
    }

    return (continue_after_parse);
}

} // namespace scm
