
#include "core.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <scm/log.h>
#include <scm/time.h>
#include <scm/core/version.h>
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

    scm::out() << version_string() << std::endl;
    scm::out() << "startup time: "
               << time::universal_time() << std::endl;
    scm::out() << log_level(logging::ll_info)
               << "initializing scm.core:" << std::endl;

    if (!initialize(argc, argv)) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::core(): errors during core initialization" << std::endl;
        shutdown();
        throw std::runtime_error("core::core: <fatal> errors during initialization");
    }

    scm::out() << log_level(logging::ll_info)
               << "successfully initialized scm.core:" << std::endl;

    scm::out() << scm::logging::core::get() << std::endl;

}

core::~core()
{
    scm::out() << log_level(logging::ll_info)
               << "shutting down scm.core:" << std::endl;

    if (!shutdown()) {
        throw std::runtime_error("core::~core: <fatal> errors during shutdown");
    }

    scm::out() << log_level(logging::ll_info)
               << "successfully shut down scm.core" << std::endl;
    scm::out() << log_level(logging::ll_info)
               << "shutdown time: "
               << time::universal_time() << std::endl;
    scm::out() << "bye sick, sad world..." << std::endl;

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

const core::command_line_option_descr&
core::command_line_options() const
{
    return (_command_line_options);
}

void
core::add_command_line_options(const core::command_line_option_descr&   opt,
                               const std::string&                       module)
{
    command_line_descr_container::iterator mod = _module_options.find(module);
    if (mod != _module_options.end()) {
        // the options for this module are allready registered
        mod->second.add(opt);
    }
    else {
        _module_options.insert(command_line_descr_container::value_type(module, opt));
    }

    _command_line_options.add(opt);
}

bool
core::initialize(int argc, char **argv)
{
    scm::out() << log_level(logging::ll_info)
               << " - running pre core init functions" << std::endl;

    _system_state = ss_pre_init;
    if (!module::initializer::run_pre_core_init_functions(instance())) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::initialize(): errors in pre core init functions" << std::endl;
        return (false);
    }

    // ok initialize the core
    _system_state = ss_init;

    _command_line_options.add_options()
            ("help", "show this help message")
            ("width", boost::program_options::value<int>()->default_value(1024), "output width")
            ("height", boost::program_options::value<int>()->default_value(640), "output height")
            ("fullscreen", boost::program_options::value<bool>()->default_value(false), "run in fullscreen mode");

    scm::out() << log_level(logging::ll_info)
               << " - parsing command line options" << std::endl;
    if (!parse_command_line(argc, argv)) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::initialize(): error parsing command line options" << std::endl;
        return (false);
    }

    scm::out() << log_level(logging::ll_info)
               << " - running post core init functions" << std::endl;

    if (!module::initializer::run_post_core_init_functions(instance())) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::initialize(): errors in post core init functions" << std::endl;
        return (false);
    }

    _system_state = ss_running;

    return (true);
}

bool
core::shutdown()
{
    _system_state = ss_shutdown;

    scm::out() << log_level(logging::ll_info)
               << " - running pre core shutdown functions" << std::endl;

    if (!module::initializer::run_pre_core_shutdown_functions(instance())) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::shutdown(): errors in pre core shutdown functions" << std::endl;
        return (false);
    }

    // shutdown core

    scm::out() << log_level(logging::ll_info)
               << " - running post core shutdown functions" << std::endl;

    if (!module::initializer::run_post_core_shutdown_functions(instance())) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::shutdown(): errors in post core shutdown functions" << std::endl;
        return (false);
    }

    return (true);
}

void
core::setup_logging(const std::string& log_file)
{
    if (!log_file.empty()) {
        try {
            logging::logger::listener_ptr   logfile_list(new logging::listener_file(log_file));
            logfile_list->style(scm::logging::listener::log_full_decorated);
            logging::core::get().default_log().add_listener(logfile_list);
        }
        catch (std::exception& e) {
            std::cerr << "core::setup_logging(): <error> " << e.what() << std::endl;
        }
#if SCM_DEBUG
        logging::core::get().default_log().log_level(logging::ll_info);
#else
        logging::core::get().default_log().log_level(logging::ll_error);
#endif
    }

    logging::logger::listener_ptr   cout_list(new logging::listener_ostream(std::cout));
    cout_list->style(scm::logging::listener::log_plain);
    logger("scm.out").log_level(logging::ll_output);
    logger("scm.out").add_listener(cout_list);

#if SCM_DEBUG
    logger("scm.err").log_level(logging::ll_debug);
#else
    logger("scm.err").log_level(logging::ll_warning);
#endif
    logging::logger::listener_ptr   cerr_list(new logging::listener_ostream(std::cerr));
    logger("scm.err").add_listener(cerr_list);
}

void
core::cleanup_logging()
{
    logger("scm.out").clear_listeners();
    logger("scm.err").clear_listeners();
    logging::core::get().default_log().clear_listeners();
}

bool
core::parse_command_line(int argc, char **argv)
{
    bool continue_after_parse = true;

    try {
        namespace bpo = boost::program_options;
        
        bpo::parsed_options      parsed_cmd_line =  bpo::parse_command_line(argc, argv, _command_line_options);

        bpo::store(parsed_cmd_line, _command_line);
        bpo::notify(_command_line);
    }
    catch (std::exception& e) {
        scm::err() << log_level(logging::ll_fatal)
                   << "core::parse_command_line(): error parsing command line (" << e.what() << ")" << std::endl;
        // print out usage
        scm::err() << log_level(logging::ll_fatal)
                   << "usage: " << std::endl
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

            command_line_descr_container::iterator mod_iter = _module_options.find(mod);
            if (mod_iter != _module_options.end()) {
                cout << "module '" << mod << "' usage: " << std::endl;
                cout << mod_iter->second << std::endl;
            }
            else {
                cout << "unknown module '" << mod << "' in the --help-module option" << std::endl;
            }
            continue_after_parse = false;
        }
    }

    return (continue_after_parse);
}

} // namespace scm
