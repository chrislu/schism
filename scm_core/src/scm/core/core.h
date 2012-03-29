
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_H_INCLUDED
#define SCM_CORE_H_INCLUDED

#include <map>
#include <string>

#include <boost/utility.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

#include <scm/core/utilities/singleton.h>
#include <scm/core/module/initializer.h>

namespace scm {

class __scm_export(core) core : boost::noncopyable
{
public:
    enum system_state_type {
        ss_undefined    = 0x00,
        ss_pre_init     = 0x10,
        ss_init         = 0x20,
        ss_running      = 0x30,
        ss_shutdown     = 0xff
    };

public:
    typedef boost::program_options::options_description             command_line_option_desc;
    typedef boost::program_options::positional_options_description  command_line_position_desc;
    typedef boost::program_options::variables_map                   command_line_result;

    typedef std::map<std::string, command_line_option_desc>         command_line_desc_container;

public:
    core(int argc, char **argv/*, app_type t = gui_application || console_application || etc.*/);
    virtual ~core();

    static bool                             check_instance();
    static core&                            instance();

    system_state_type                       system_state() const;

    std::string                             version_string() const;

    const command_line_option_desc&         command_line_options() const;
    void                                    add_command_line_options(const command_line_option_desc& opt,
                                                                     const std::string& module);
    command_line_position_desc&             command_line_positions();

protected:
    bool                                    initialize(int argc, char **argv);
    bool                                    shutdown();

    void                                    setup_logging(const std::string& log_file);
    void                                    cleanup_logging();

    bool                                    parse_command_line(int argc, char **argv);

protected:
    system_state_type                       _system_state;

    command_line_option_desc                _command_line_options;
    command_line_position_desc              _command_line_positions;
    command_line_desc_container             _module_options;
    command_line_result                     _command_line;

private:
    static core*                            _instance;

}; //  class core

} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_H_INCLUDED
